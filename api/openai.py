import asyncio
from http import HTTPStatus
import json
import time
from fastapi import APIRouter, Depends, Request, Header, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, StreamingResponse
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

        if body_data.get("stream", False):
            # Return streaming response
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
        else:
            # Return regular JSON response
            # Convert the response to a dictionary if it's not already
            if hasattr(response, "dict"):
                response_data = response.dict()
            else:
                response_data = response

            # Update token counts for non-streaming response
            if hasattr(response, "usage"):
                total_tokens["prompt"] = getattr(response.usage, "prompt_tokens", 0)
                total_tokens["completion"] = getattr(
                    response.usage, "completion_tokens", 0
                )

                # Record token usage
                token_usage_queue = request.app.state.token_usage_queue
                await token_usage_queue.put(
                    (
                        model_id,
                        response.id if hasattr(response, "id") else "unknown",
                        api_key_id,
                        total_tokens["prompt"],
                        total_tokens["completion"],
                        "completed",
                    )
                )

            return JSONResponse(response_data)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


class ModelResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]


# 请求部署，并且匹配到目标模型返回
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
        # Start a transaction
        async with conn.transaction():
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

            # Get all associated model records (including inactive ones)
            records = await conn.fetch(
                """
                SELECT 
                    im.model_name,
                    im.model_id,
                    im.id as model_db_id,
                    im.status as model_status,
                    id.deployment_url,
                    id.models_api_key,
                    id.id as deployment_id,
                    id.status as deployment_status
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
            # For each record, fetch models from the deployment URL
            async with httpx.AsyncClient() as client:
                tasks = []
                for r in records:
                    if not r["deployment_url"] or not r["models_api_key"]:
                        # Mark model as inactive if URL or API key is missing
                        if r["model_status"] != "inactive":
                            await conn.execute(
                                "UPDATE inference_model SET status = 'inactive' WHERE id = $1",
                                r["model_db_id"],
                            )
                        continue

                    tasks.append(fetch_models_for_record(client, r, conn))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        continue
                    all_models.extend(result)

            return {"object": "list", "data": all_models}


async def fetch_models_for_record(
    client: httpx.AsyncClient, record: Dict, conn
) -> List[Dict]:
    """Fetch and process model data for a single record, updating status if needed"""
    url = f"{record['deployment_url'].rstrip('/')}/v1/models"
    headers = {"Authorization": f"Bearer {record['models_api_key']}"}
    matched_models = []

    try:
        response = await client.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        models_data = response.json()

        has_matching_model = False
        if "data" in models_data:
            for model in models_data["data"]:
                if model.get("id", "").endswith(record["model_id"]):
                    # Replace id and root with our model_name
                    model["id"] = record["model_name"]
                    if "root" in model:
                        model["root"] = record["model_name"]
                    matched_models.append(model)
                    has_matching_model = True

        # Update status based on whether we found a matching model
        if has_matching_model:
            # If we found a match and the model was previously inactive, reactivate it
            if record["model_status"] == "inactive":
                await conn.execute(
                    "UPDATE inference_model SET status = 'active' WHERE id = $1",
                    record["model_db_id"],
                )
            # Also check deployment status
            if record["deployment_status"] == "inactive":
                await conn.execute(
                    "UPDATE inference_deployment SET status = 'active' WHERE id = $1",
                    record["deployment_id"],
                )
        else:
            # If no matching model was found in the response, mark as inactive
            if record["model_status"] != "inactive":
                await conn.execute(
                    "UPDATE inference_model SET status = 'inactive' WHERE id = $1",
                    record["model_db_id"],
                )

    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        print(f"Error fetching models from {url}: {str(e)}")
        # Mark both the model and deployment as inactive if they weren't already
        if record["model_status"] != "inactive":
            await conn.execute(
                "UPDATE inference_model SET status = 'inactive' WHERE id = $1",
                record["model_db_id"],
            )
        if record["deployment_status"] != "inactive":
            await conn.execute(
                "UPDATE inference_deployment SET status = 'inactive' WHERE id = $1",
                record["deployment_id"],
            )
    except Exception as e:
        print(f"Unexpected error for {url}: {str(e)}")
        if record["model_status"] != "inactive":
            await conn.execute(
                "UPDATE inference_model SET status = 'inactive' WHERE id = $1",
                record["model_db_id"],
            )

    return matched_models


async def generate_response(
    request, response, encoder, total_tokens, model_id, current_model, api_key_id
) -> AsyncGenerator[str, None]:
    try:
        last_chunk_id = None
        for chunk in response:
            # print("[DEBUG] Chunk:", chunk)
            last_chunk_id = getattr(chunk, "id", "unknown")

            # 优先使用 usage 记录token
            if hasattr(chunk, "usage") and chunk.usage is not None:
                total_tokens["prompt"] = getattr(chunk.usage, "prompt_tokens", 0)
                total_tokens["completion"] = getattr(
                    chunk.usage, "completion_tokens", 0
                )
                print(f"Total Prompt Tokens: {total_tokens['prompt']}")
                print(f"Total Completion Tokens: {total_tokens['completion']}")

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
                        last_chunk_id,
                        api_key_id,
                        total_tokens["prompt"],
                        total_tokens["completion"],
                        "interrupted",
                    )
                )
                return

            chunk_data = {
                "id": last_chunk_id,
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
                # 当流结束时记录token使用情况
                token_usage_queue = request.app.state.token_usage_queue
                await token_usage_queue.put(
                    (
                        model_id,
                        last_chunk_id,
                        api_key_id,
                        total_tokens["prompt"],
                        total_tokens["completion"],
                        "completed",
                    )
                )
                yield "data: [DONE]\n\n"

    except Exception as e:
        print(f"Error in generate_response: {e}")
        if last_chunk_id:
            token_usage_queue = request.app.state.token_usage_queue
            await token_usage_queue.put(
                (
                    model_id,
                    last_chunk_id,
                    api_key_id,
                    total_tokens["prompt"],
                    total_tokens["completion"],
                    "error",
                )
            )
        raise
