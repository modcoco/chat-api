import asyncio
from http import HTTPStatus
import json
import time
from fastapi import APIRouter, Depends, Request, Header, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
import httpx
import tiktoken
from typing import Any, AsyncGenerator, Dict, List, Optional
from datetime import datetime, timezone

from app.inference_api_key import get_api_key_id_by_key, update_last_used_at
from app.inference_deployment import get_deployment_info_by_api_key
from app.inference_model import get_model_info_by_api_key_and_model_name
from app.inference_usage import check_api_key_usage
from app.openai_client import get_client
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pydantic import BaseModel

from protocol import ChatCompletionRequest, StreamOptions

router = APIRouter()
encoder = tiktoken.get_encoding("cl100k_base")


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise HTTPException(
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported Media Type: Only 'application/json' is allowed",
        )


# https://github.com/vllm-project/vllm/blob/f90a37559315defd369441c4d2461989a10b9fc1/vllm/entrypoints/openai/api_server.py#L399
@router.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def proxy_openai(
    request: Request, body: ChatCompletionRequest, authorization: str = Header(None)
):
    db = request.app.state.db_pool
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

    # Update last used at
    await update_last_used_at(api_key, db)

    # Use Model
    current_model = body.model
    print(f"Use Model: {body.model}")
    model_info = await get_model_info_by_api_key_and_model_name(api_key, body.model, db)
    if model_info is None:
        print("The model cannot be found.")
        raise HTTPException(
            status_code=400,
            detail="The model cannot be found.",
        )

    model_id, model_path = model_info

    # 检查api-key配额, todo: cache
    apikey_check_res = await check_api_key_usage(model_id, api_key, db, True)
    if apikey_check_res is not None:
        print("The usage is over the quota:", api_key, apikey_check_res)
        raise HTTPException(
            status_code=400,
            detail=apikey_check_res,
        )

    total_tokens = {"prompt": 0, "completion": 0}
    extracted_content = []
    for msg in body.messages:
        content = msg.get("content")
        if isinstance(content, str):
            extracted_content.append(content)
            tokens = len(encoder.encode(content))
            total_tokens["prompt"] += tokens
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text = part["text"]
                    extracted_content.append(text)
                    tokens = len(encoder.encode(text))
                    total_tokens["completion"] += tokens

    # Get active inference deployment
    deployment_info = await get_deployment_info_by_api_key(api_key, model_id, db)
    if deployment_info is None:
        print("The inference deployment is down.")
        raise HTTPException(
            status_code=400,
            detail="The inference deployment is down",
        )
    try:
        deployment_url, models_api_key = deployment_info

        print(deployment_info)
        client = get_client(deployment_url + "/v1", models_api_key)
        body.model = model_path
        # body.temperature = 0
        # body.stream = True
        # body.stream_options = StreamOptions(include_usage=True)
        body_data = {
            key: value for key, value in body.model_dump().items() if value is not None
        }
        print("body_data", body_data)
        response = client.chat.completions.create(**body_data)

        return StreamingResponse(
            generate_response(
                request,
                response,
                encoder,
                total_tokens,
                model_id,
                current_model,
                api_key_id,
            ),
            media_type="text/event-stream",
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


class ModelResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]


@router.get("/v1/models", response_model=ModelResponse)
async def get_models_by_api_key(
    request: Request, authorization: str = Depends(api_key_header)
):
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'",
        )

    api_key = authorization[7:].strip()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty API key",
        )

    async with request.app.state.db_pool.acquire() as conn:
        api_key_record = await conn.fetchrow(
            """
            SELECT id FROM inference_api_key 
            WHERE api_key = $1 AND is_deleted = FALSE 
            AND (expires_at IS NULL OR expires_at > NOW())
            """,
            api_key,
        )
        if not api_key_record:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or expired API Key",
            )

        # 获取所有关联的模型记录
        records = await conn.fetch(
            """
            SELECT 
                im.model_name,
                im.model_id,
                id.deployment_url,
                id.models_api_key
            FROM inference_api_key_model iakm
            JOIN inference_model im ON iakm.model_id = im.id
            JOIN inference_deployment id ON im.inference_id = id.id
            WHERE iakm.api_key_id = $1 
            AND iakm.is_deleted = FALSE
            AND im.is_deleted = FALSE 
            AND id.is_deleted = FALSE
            """,
            api_key_record["id"],
        )

        all_models = []
        # 为每个record并行请求对应的/v1/models接口
        async with httpx.AsyncClient() as client:
            tasks = []
            for r in records:
                if not r["deployment_url"] or not r["models_api_key"]:
                    continue

                tasks.append(fetch_models_for_record(client, r))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    continue
                all_models.extend(result)

        return {"object": "list", "data": all_models}


async def fetch_models_for_record(
    client: httpx.AsyncClient, record: Dict
) -> List[Dict]:
    """为单个record获取并处理模型数据"""
    try:
        url = f"{record['deployment_url'].rstrip('/')}/v1/models"
        headers = {"Authorization": f"Bearer {record['models_api_key']}"}
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        models_data = response.json()

        matched_models = []
        if "data" in models_data:
            for model in models_data["data"]:
                if model.get("id", "").endswith(record["model_id"]):
                    # 替换id和root为数据库中的model_name
                    model["id"] = record["model_name"]
                    if "root" in model:
                        model["root"] = record["model_name"]
                    matched_models.append(model)

        return matched_models

    except httpx.HTTPStatusError as e:
        print(f"HTTP error for {url}: {e.response.status_code}")
    except Exception as e:
        print(f"Error fetching models from {url}: {str(e)}")

    return []


async def generate_response(
    request, response, encoder, total_tokens, model_id, current_model, api_key_id
) -> AsyncGenerator[str, None]:
    try:
        for chunk in response:
            print("Chunk:", chunk)

            # 处理元组形式的chunk
            if isinstance(chunk, tuple) and len(chunk) >= 2:
                chunk_id = chunk[1] if chunk[0] == "id" else None
                # 创建一个简单的模拟对象
                chunk = type(
                    "SimpleChunk",
                    (),
                    {
                        "id": chunk_id,
                        "usage": None,
                        "choices": [],
                        "object": "chat.completion.chunk",
                        "delta": type("Delta", (), {"content": None}),
                    },
                )()

            # 处理usage
            if hasattr(chunk, "usage") and chunk.usage is not None:
                total_tokens["prompt"] = getattr(chunk.usage, "prompt_tokens", 0)
                total_tokens["completion"] = getattr(
                    chunk.usage, "completion_tokens", 0
                )
                print(f"Total Prompt Tokens: {total_tokens['prompt']}")
                print(f"Total Completion Tokens: {total_tokens['completion']}")

                token_usage_queue = request.app.state.token_usage_queue
                await token_usage_queue.put(
                    (
                        model_id,
                        chunk.id,
                        api_key_id,
                        total_tokens["prompt"],
                        total_tokens["completion"],
                        "completed",
                    )
                )
                break

            # 处理choices
            if hasattr(chunk, "choices"):
                for choice in chunk.choices:
                    if (
                        hasattr(choice, "delta")
                        and hasattr(choice.delta, "content")
                        and choice.delta.content
                    ):
                        total_tokens["completion"] += len(
                            encoder.encode(choice.delta.content)
                        )

            if await request.is_disconnected():
                print("客户端主动断开连接")
                print(f"Total Prompt Tokens (手动计算): {total_tokens['prompt']}")
                print(
                    f"Total Completion Tokens (手动计算): {total_tokens['completion']}"
                )

                token_usage_queue = request.app.state.token_usage_queue
                await token_usage_queue.put(
                    (
                        model_id,
                        chunk.id if hasattr(chunk, "id") else "unknown",
                        api_key_id,
                        total_tokens["prompt"],
                        total_tokens["completion"],
                        "interrupted",
                    )
                )
                return

            # 构造 chunk_data
            chunk_data = {
                "id": getattr(chunk, "id", "unknown"),
                "object": getattr(chunk, "object", "chat.completion.chunk"),
                "created": int(time.time()),
                "model": current_model,
                "choices": [
                    {
                        "index": getattr(choice, "index", 0),
                        "delta": {
                            "content": getattr(
                                getattr(choice, "delta", None), "content", None
                            ),
                            "function_call": getattr(
                                getattr(choice, "delta", None), "function_call", None
                            ),
                            "refusal": getattr(
                                getattr(choice, "delta", None), "refusal", None
                            ),
                            "role": getattr(
                                getattr(choice, "delta", None), "role", None
                            ),
                            "tool_calls": getattr(
                                getattr(choice, "delta", None), "tool_calls", None
                            ),
                        },
                        "finish_reason": getattr(choice, "finish_reason", None),
                        "stop_reason": getattr(choice, "stop_reason", None),
                    }
                    for choice in getattr(chunk, "choices", [])
                ],
                "usage": getattr(chunk, "usage", None),
            }

            # 转换为 JSON 字符串
            chunk_json = json.dumps(
                chunk_data,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            yield f"data: {chunk_json}\n\n"

            if hasattr(chunk, "choices") and any(
                getattr(choice, "finish_reason", None) == "stop"
                for choice in chunk.choices
            ):
                yield "data: [DONE]\n\n"

    except Exception as e:
        print(f"Error in generate_response: {e}")
        raise
