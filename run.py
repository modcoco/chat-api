import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import os
import time
import json
from typing import Dict, List, Optional, Tuple
import uuid
import asyncpg
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from grpc import Status
import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
import tiktoken

from inference_deployment import create_inference_deployment, get_inference_deployments
from inference_model import create_inference_model
from models import (
    InferenceDeployment,
    InferenceDeploymentCreate,
    InferenceModelApiKeyCreate,
    InferenceModelApiKeyResponse,
    InferenceModelCreate,
)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_user = os.getenv("DB_USER", "default_user")
    db_password = os.getenv("DB_PASSWORD", "default_password")
    db_name = os.getenv("DB_NAME", "postgres")
    db_host = os.getenv("DB_HOST", "localhost")
    db_min_size = int(os.getenv("DB_MIN_SIZE", "1"))
    db_max_size = int(os.getenv("DB_MAX_SIZE", "10"))

    print("Initializing database connection pool...")
    pool = await asyncpg.create_pool(
        user=db_user,
        password=db_password,
        database=db_name,
        host=db_host,
        min_size=db_min_size,
        max_size=db_max_size,
    )
    app.state.db_pool = pool

    app.state.token_usage_queue = asyncio.Queue()
    asyncio.create_task(process_token_usage_queue(app))

    yield
    await pool.close()
    print("Closing database connection pool...")


async def process_token_usage_queue(app: FastAPI):
    while True:
        token_data = await app.state.token_usage_queue.get()  # 获取队列中的数据
        if token_data is None:
            break

        (
            completions_chunk_id,
            api_key_id,
            prompt_tokens,
            completion_tokens,
            type,
        ) = token_data
        try:
            await insert_token_usage(
                app.state.db_pool,
                completions_chunk_id,
                api_key_id,
                prompt_tokens,
                completion_tokens,
                type,
            )
        except Exception as e:
            print(f"数据库插入失败: {e}")
        finally:
            app.state.token_usage_queue.task_done()  # 标记任务完成


app = FastAPI(lifespan=lifespan)

# 客户端连接池
client_pool: Dict[str, dict] = {}
# 客户端超时时间（30 分钟）
CLIENT_TIMEOUT = 30 * 60


def get_client(base_url: str, api_key: str) -> OpenAI:
    """
    获取或创建客户端实例，并支持超时清理
    """
    global client_pool

    cleanup_clients()

    # 如果客户端已存在且未超时，则直接返回
    if base_url in client_pool:
        client_data = client_pool[base_url]
        if time.time() - client_data["last_used"] <= CLIENT_TIMEOUT:
            client_data["last_used"] = time.time()  # 更新最后使用时间
            return client_data["client"]

    # 如果客户端不存在或已超时，则创建新的客户端
    client = OpenAI(base_url=base_url, api_key=api_key)
    client_pool[base_url] = {
        "client": client,
        "last_used": time.time(),  # 记录最后使用时间
    }
    return client


def cleanup_clients():
    """
    清理超时的客户端
    """
    global client_pool
    current_time = time.time()
    for base_url in list(client_pool.keys()):
        client_data = client_pool[base_url]
        if current_time - client_data["last_used"] > CLIENT_TIMEOUT:
            del client_pool[base_url]


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

encoder = tiktoken.get_encoding("cl100k_base")


@app.post("/v1/chat/completions")
async def proxy_openai(
    request: Request,
    authorization: str = Header(None),
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication Fail",
        )
    api_key = authorization.split("Bearer ")[-1]
    print(f"Received API Key: {api_key}")

    api_key_id = await get_api_key_id_by_key(api_key, db)
    if api_key_id is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication Fail",
        )

    # 检查api-key配额, todo: cache
    apikey_check_res = await check_api_key_usage(api_key, db, True)
    if apikey_check_res is not None:
        print("The usage is over the quota:", api_key, apikey_check_res)
        raise HTTPException(
            status_code=400,
            detail=apikey_check_res,
        )

    # 获取模型
    user_request = await request.json()
    # Init total_tokens
    messages = user_request.get("messages", [])
    model_name = user_request.get("model")
    prompt_text = " ".join([msg.get("content", "") for msg in messages])
    total_tokens = {"prompt": len(encoder.encode(prompt_text)), "completion": 0}

    # Use Model
    print(f"Use Model: {model_name}")
    model_id = await get_model_id_by_api_key_and_model_name(api_key, model_name, db)
    if model_id is None:
        print("The model cannot be found.")
        raise HTTPException(
            status_code=400,
            detail="The model cannot be found.",
        )

    # Get active inference deployment
    deployment_info = await get_deployment_info_by_api_key(api_key, db)
    if deployment_info is None:
        print("The inference deployment is down.")
        raise HTTPException(
            status_code=400,
            detail="The inference deployment is down",
        )
    try:
        deployment_url, models_api_key = deployment_info
        client1 = get_client(deployment_url + "/v1", models_api_key)
        response = client1.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )

        return StreamingResponse(
            generate_response(
                request, response, app, encoder, total_tokens, api_key_id
            ),
            media_type="text/event-stream",
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


async def generate_response(request, response, app, encoder, total_tokens, api_key_id):
    try:
        for chunk in response:
            if chunk.usage is not None:
                total_tokens["prompt"] = chunk.usage.prompt_tokens
                total_tokens["completion"] = chunk.usage.completion_tokens
                print(f"Total Prompt Tokens: {total_tokens['prompt']}")
                print(f"Total Completion Tokens: {total_tokens['completion']}")

                await app.state.token_usage_queue.put(
                    (
                        chunk.id,
                        api_key_id,  # api-key-id
                        total_tokens["prompt"],
                        total_tokens["completion"],
                        "completed",
                    )
                )
                break

            for choice in chunk.choices:
                if choice.delta.content:
                    total_tokens["completion"] += len(
                        encoder.encode(choice.delta.content)
                    )

            if await request.is_disconnected():
                print("客户端主动断开连接")
                print(f"Total Prompt Tokens (手动计算): {total_tokens['prompt']}")
                print(
                    f"Total Completion Tokens (手动计算): {total_tokens['completion']}"
                )

                await app.state.token_usage_queue.put(
                    (
                        chunk.id,
                        api_key_id,  # api-key-id
                        total_tokens["prompt"],
                        total_tokens["completion"],
                        "interrupted",
                    )
                )

                return

            chunk_data = ChatCompletionChunk(
                id=chunk.id,
                object=chunk.object,
                created=int(time.time()),
                model=chunk.model,
                choices=[
                    Choice(
                        index=choice.index,
                        delta=ChoiceDelta(
                            content=choice.delta.content,
                            function_call=choice.delta.function_call,
                            refusal=choice.delta.refusal,
                            role=choice.delta.role,
                            tool_calls=choice.delta.tool_calls,
                        ),
                        finish_reason=choice.finish_reason,
                        stop_reason=getattr(choice, "stop_reason", None),
                    )
                    for choice in chunk.choices
                ],
                usage=chunk.usage,
            )

            chunk_json = json.dumps(
                chunk_data.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            yield f"data: {chunk_json}\n\n".encode("utf-8")

            if chunk.choices and any(
                choice.finish_reason == "stop" for choice in chunk.choices
            ):
                yield "data: [DONE]\n\n".encode("utf-8")
    except Exception as e:
        print(f"Error in generate_response: {e}")
        raise


@app.get("/v1/models")
async def get_users():
    async with app.state.db_pool.acquire() as connection:
        result = await connection.fetch("SELECT * FROM models")
        return {"users": [dict(record) for record in result]}


@app.post("/inference-deployment")
async def create_deployment(
    deployment: InferenceDeploymentCreate,
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
    async with db.acquire() as conn:
        existing_deployment = await conn.fetchrow(
            "SELECT * FROM inference_deployment WHERE inference_name = $1",
            deployment.inference_name,
        )
        if existing_deployment:
            raise HTTPException(
                status_code=400, detail="Inference deployment already exists."
            )

        new_deployment = await create_inference_deployment(conn, deployment)
        return new_deployment


@app.get("/inference-deployment", response_model=List[InferenceDeployment])
async def get_deployments(
    inference_name: Optional[str] = None,
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
    async with db.acquire() as conn:
        deployments = await get_inference_deployments(conn, inference_name)
        if not deployments:
            raise HTTPException(
                status_code=404, detail="Inference deployments not found."
            )
        return deployments


@app.delete("/inference-deployment/{id}", status_code=204)
async def delete_deployment(
    id: int,
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
    async with db.acquire() as conn:
        existing_deployment = await conn.fetchrow(
            "SELECT * FROM inference_deployment WHERE id = $1", id
        )
        if not existing_deployment:
            raise HTTPException(
                status_code=404, detail="Inference deployment not found."
            )

        await conn.execute("DELETE FROM inference_deployment WHERE id = $1", id)

        return {"detail": "Inference deployment deleted successfully."}


@app.get("/inference-deployment/{id}/models", response_model=List[dict])
async def get_model_ids(id: int, db: asyncpg.Pool = Depends(lambda: app.state.db_pool)):
    async with db.acquire() as conn:
        deployment = await conn.fetchrow(
            "SELECT deployment_url FROM inference_deployment WHERE id = $1", id
        )

        if not deployment:
            raise HTTPException(
                status_code=404, detail="Inference deployment not found."
            )

        deployment_url = deployment["deployment_url"]
        url_with_models = f"{deployment_url}/v1/models"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url_with_models)
                response.raise_for_status()

                data = response.json()

                if "data" in data:
                    filtered_data = [
                        {
                            "id": item.get("id"),
                            "inference_id": id,
                            "created": item.get("created"),
                            "owned_by": item.get("owned_by"),
                            "max_model_len": item.get("max_model_len"),
                        }
                        for item in data["data"]
                    ]
                    return filtered_data
                else:
                    return []
            except httpx.RequestError as e:
                return []
            except httpx.HTTPStatusError as e:
                return []


# API
@app.post("/inference-model", response_model=InferenceModelCreate)
async def add_inference_model(
    model: InferenceModelCreate, db: asyncpg.Pool = Depends(lambda: app.state.db_pool)
):
    async with db.acquire() as conn:
        inserted_model = await create_inference_model(conn, model)
        return JSONResponse(content=inserted_model, status_code=201)


@app.get("/inference-model", response_model=List[dict])
async def get_all_inference_services(
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
    query = """
    SELECT 
        im.id,
        im.model_name,
        im.visibility,
        im.user_id,
        im.team_id,
        im.inference_id,
        im.model_id,
        im.max_token_quota,
        im.max_prompt_tokens_quota,
        im.max_completion_tokens_quota,
        im.created,
        im.updated
    FROM inference_model im
    JOIN inference_deployment idp ON im.inference_id = idp.id;
    """

    async with db.acquire() as conn:
        services = await conn.fetch(query)
        return [dict(service) for service in services]


@app.delete("/inference-model/{id}", response_model=dict)
async def delete_inference_model(
    id: int, db: asyncpg.Pool = Depends(lambda: app.state.db_pool)
):
    query = "DELETE FROM inference_model WHERE id = $1 RETURNING id, model_name, visibility, user_id, team_id, inference_id, model_id, max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, created, updated;"

    async with db.acquire() as conn:
        result = await conn.fetchrow(query, id)

        if not result:
            raise HTTPException(status_code=404, detail="Inference model not found")

        return dict(result)


@app.post("/api-key", response_model=dict)
async def create_inference_model_api_key(
    api_key_data: InferenceModelApiKeyCreate,
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
    unique_id = uuid.uuid4().hex
    api_key = f"sk-{unique_id}"

    if api_key_data.active_days:
        expires_at = datetime.now() + timedelta(days=api_key_data.active_days)
    else:
        expires_at = None

    created_time = datetime.now()

    query = """
    INSERT INTO inference_model_api_key (
        user_id, api_key_name, inference_model_id, api_key, 
        max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, 
        created, expires_at,active_days
    ) 
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) 
    RETURNING id, user_id, api_key_name, inference_model_id, api_key, 
              max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, 
              created, expires_at, active_days, is_deleted;
    """

    async with db.acquire() as conn:
        result = await conn.fetchrow(
            query,
            api_key_data.user_id,
            api_key_data.api_key_name,
            api_key_data.inference_model_id,
            api_key,
            api_key_data.max_token_quota,
            api_key_data.max_prompt_tokens_quota,
            api_key_data.max_completion_tokens_quota,
            created_time,
            expires_at,
            api_key_data.active_days,
        )

    return dict(result)


@app.get("/api-keys", response_model=List[InferenceModelApiKeyResponse])
async def get_inference_model_api_keys(
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
    query = """
    SELECT id, user_id, api_key_name, inference_model_id, api_key, 
           max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota,
           active_days, created, last_used_at, expires_at, is_deleted
    FROM inference_model_api_key
    WHERE is_deleted = FALSE;
    """

    async with db.acquire() as conn:
        result = await conn.fetch(query)

    return [
        {
            "id": row["id"],
            "user_id": row["user_id"],
            "api_key_name": row["api_key_name"],
            "inference_model_id": row["inference_model_id"],
            "api_key": row["api_key"],
            "max_token_quota": row.get("max_token_quota"),
            "max_prompt_tokens_quota": row.get("max_prompt_tokens_quota"),
            "max_completion_tokens_quota": row.get("max_completion_tokens_quota"),
            "active_days": row.get("active_days"),
            "created": row["created"].isoformat(),
            "last_used_at": (
                row["last_used_at"].isoformat() if row["last_used_at"] else None
            ),
            "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
            "is_deleted": row["is_deleted"],
        }
        for row in result
    ]


@app.patch("/api-key/{api_key_id}/delete")
async def delete_inference_model_api_key(
    api_key_id: int,
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
    query = """
    SELECT id FROM inference_model_api_key WHERE id = $1 AND is_deleted = FALSE;
    """
    async with db.acquire() as conn:
        result = await conn.fetchrow(query, api_key_id)

    if not result:
        raise HTTPException(
            status_code=Status.HTTP_404_NOT_FOUND,
            detail="API key not found or already deleted",
        )

    update_query = """
    UPDATE inference_model_api_key
    SET is_deleted = TRUE
    WHERE id = $1
    RETURNING id, is_deleted;
    """

    async with db.acquire() as conn:
        updated_result = await conn.fetchrow(update_query, api_key_id)

    return {"id": updated_result["id"], "is_deleted": updated_result["is_deleted"]}


async def insert_token_usage(
    db_pool,
    completions_chunk_id,
    api_key_id,
    prompt_tokens,
    completion_tokens,
    type,
):
    """
    插入 token 使用记录到数据库
    """
    retries = 3
    for attempt in range(retries):
        try:
            async with db_pool.acquire() as connection:
                async with connection.transaction():
                    await connection.execute(
                        """
                        INSERT INTO inference_model_api_key_token_usage 
                        (completions_chunk_id, api_key_id, prompt_tokens, completion_tokens, type, created)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        completions_chunk_id,
                        api_key_id,
                        prompt_tokens,
                        completion_tokens,
                        type,
                        datetime.fromtimestamp(time.time()),
                    )
            print("数据插入成功")
            break
        except Exception as db_error:
            if attempt == retries - 1:
                print(f"数据库插入失败，重试 {attempt + 1} 次后仍失败: {db_error}")
                raise
            else:
                print(f"数据库插入失败，正在重试 ({attempt + 1}/{retries}): {db_error}")
                await asyncio.sleep(1)


async def get_api_key_id_by_key(api_key: str, db: asyncpg.Pool) -> Optional[int]:
    query = """
    SELECT id FROM inference_model_api_key WHERE api_key = $1 AND is_deleted = FALSE;
    """

    async with db.acquire() as conn:
        result = await conn.fetchrow(query, api_key)

    if result:
        return result["id"]
    return None


# check_model_quota 是否检查模型配哦额
async def check_api_key_usage(
    api_key: str, db: asyncpg.Pool, check_model_quota: bool = True
) -> Optional[str]:
    # 获取 api_key 对应的配额信息
    query = """
    SELECT max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, inference_model_id
    FROM inference_model_api_key
    WHERE api_key = $1 AND is_deleted = FALSE;
    """

    async with db.acquire() as conn:
        result = await conn.fetchrow(query, api_key)

    if not result:
        return "API Key not found or is deleted"

    max_token_quota = result["max_token_quota"]
    max_prompt_tokens_quota = result["max_prompt_tokens_quota"]
    max_completion_tokens_quota = result["max_completion_tokens_quota"]
    inference_model_id = result["inference_model_id"]

    # 获取该 api_key 的所有 token 使用记录
    usage_query = """
    SELECT SUM(prompt_tokens) AS total_prompt_tokens, SUM(completion_tokens) AS total_completion_tokens
    FROM inference_model_api_key_token_usage
    WHERE api_key_id = (SELECT id FROM inference_model_api_key WHERE api_key = $1) 
    AND created <= $2;  -- 可选：限制到某个时间范围（例如当前时间前）
    """

    async with db.acquire() as conn:
        usage = await conn.fetchrow(usage_query, api_key, datetime.now())

    if not usage:
        return "No token usage records found for this API key"

    total_prompt_tokens = usage["total_prompt_tokens"] or 0
    total_completion_tokens = usage["total_completion_tokens"] or 0

    # 判断是否超出 API Key 配额
    if (
        max_prompt_tokens_quota is not None
        and total_prompt_tokens > max_prompt_tokens_quota
    ):
        return "Prompt tokens exceeded quota"

    if (
        max_completion_tokens_quota is not None
        and total_completion_tokens > max_completion_tokens_quota
    ):
        return "Completion tokens exceeded quota"

    if (
        max_token_quota is not None
        and (total_prompt_tokens + total_completion_tokens) > max_token_quota
    ):
        return "Total tokens exceeded quota"

    # 如果 API key 的配额没有超出，则进行模型配额检查
    if check_model_quota:
        model_check_res = await check_model_usage(inference_model_id, api_key, db)
        if model_check_res:
            return model_check_res

    # 如果没有超出配额
    return None


async def check_model_usage(
    inference_model_id: int, api_key: str, db: asyncpg.Pool
) -> Optional[str]:
    # 获取模型的配额信息
    model_query = """
    SELECT max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota
    FROM inference_model
    WHERE id = $1;
    """

    async with db.acquire() as conn:
        model = await conn.fetchrow(model_query, inference_model_id)

    if not model:
        return "Model not found"

    model_max_token_quota = model["max_token_quota"]
    model_max_prompt_tokens_quota = model["max_prompt_tokens_quota"]
    model_max_completion_tokens_quota = model["max_completion_tokens_quota"]

    # 获取该 api_key 对应的 inference_model_id
    api_key_query = """
    SELECT inference_model_id
    FROM inference_model_api_key
    WHERE api_key = $1 AND is_deleted = FALSE;
    """

    async with db.acquire() as conn:
        api_key_result = await conn.fetchrow(api_key_query, api_key)

    if not api_key_result:
        return "API Key not found or is deleted"

    api_key_inference_model_id = api_key_result["inference_model_id"]

    # 检查是否该api_key属于指定模型
    if api_key_inference_model_id != inference_model_id:
        return "API Key does not belong to the specified model"

    # 获取模型的使用量
    model_usage_query = """
    SELECT SUM(prompt_tokens) AS total_prompt_tokens, SUM(completion_tokens) AS total_completion_tokens
    FROM inference_model_api_key_token_usage
    WHERE api_key_id = (SELECT id FROM inference_model_api_key WHERE api_key = $1)
    AND created <= $2;
    """

    async with db.acquire() as conn:
        model_usage = await conn.fetchrow(model_usage_query, api_key, datetime.now())

    if model_usage:
        total_model_prompt_tokens = model_usage["total_prompt_tokens"] or 0
        total_model_completion_tokens = model_usage["total_completion_tokens"] or 0

        # 检查模型的配额是否超出
        if (
            model_max_prompt_tokens_quota is not None
            and total_model_prompt_tokens > model_max_prompt_tokens_quota
        ):
            return "Model's prompt tokens exceeded quota"

        if (
            model_max_completion_tokens_quota is not None
            and total_model_completion_tokens > model_max_completion_tokens_quota
        ):
            return "Model's completion tokens exceeded quota"

        if (
            model_max_token_quota is not None
            and (total_model_prompt_tokens + total_model_completion_tokens)
            > model_max_token_quota
        ):
            return "Model's total tokens exceeded quota"

    return None


async def get_model_id_by_api_key_and_model_name(
    api_key: str, model_name: str, db: asyncpg.Pool
) -> Optional[str]:
    # 使用 JOIN 查询获取 model_id 并验证 model_name 是否匹配
    query = """
    SELECT im.model_id
    FROM inference_model_api_key imak
    JOIN inference_model im ON imak.inference_model_id = im.id
    WHERE imak.api_key = $1
    AND im.model_name = $2
    AND imak.is_deleted = FALSE
    """

    async with db.acquire() as conn:
        result = await conn.fetchrow(query, api_key, model_name)

    # 如果找到匹配的结果，则返回 model_id，否则返回 None
    if result:
        return result["model_id"]
    else:
        return None


async def get_deployment_info_by_api_key(
    api_key: str, db: asyncpg.Pool
) -> Optional[Tuple[str, str]]:
    # 使用 JOIN 查询获取 deployment_url 和 models_api_key，明确指定每个表的 id 列
    query = """
    SELECT idp.deployment_url, idp.models_api_key
    FROM inference_deployment idp
    JOIN inference_model im ON idp.id = im.inference_id
    JOIN inference_model_api_key imak ON im.id = imak.inference_model_id
    WHERE imak.api_key = $1
    AND imak.is_deleted = FALSE
    """

    async with db.acquire() as conn:
        result = await conn.fetchrow(query, api_key)

    # 如果找到了匹配的结果，则返回 deployment_url 和 models_api_key，否则返回 None
    if result:
        return result["deployment_url"], result["models_api_key"]
    else:
        return None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
