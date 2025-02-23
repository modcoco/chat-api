import asyncio
from contextlib import asynccontextmanager
import datetime
import os
import time
import json
import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
import tiktoken

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


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

encoder = tiktoken.get_encoding("cl100k_base")


@app.post("/v1/chat/completions")
async def proxy_openai(request: Request, authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication Fail",
        )
    api_key = authorization.split("Bearer ")[-1]
    print(api_key)
    try:
        user_request = await request.json()
        # Init total_tokens
        messages = user_request.get("messages", [])
        prompt_text = " ".join([msg.get("content", "") for msg in messages])
        total_tokens = {"prompt": len(encoder.encode(prompt_text)), "completion": 0}

        response = client.chat.completions.create(
            model="/mnt/data/models/deepseek-ai_DeepSeek-R1",
            messages=messages,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )

        return StreamingResponse(
            generate_response(request, response, app, encoder, total_tokens),
            media_type="text/event-stream",
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


async def generate_response(request, response, app, encoder, total_tokens):
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
                        1, # api-key-id
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
                        1, # api-key-id
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
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        completions_chunk_id,
                        api_key_id,
                        prompt_tokens,
                        completion_tokens,
                        type,
                        datetime.datetime.fromtimestamp(time.time()),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
