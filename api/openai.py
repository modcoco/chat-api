from http import HTTPStatus
import json
import time
from fastapi import APIRouter, Depends, Request, Header, HTTPException
from fastapi.responses import StreamingResponse
import tiktoken

from app.inference_api_key import get_api_key_id_by_key, update_last_used_at
from app.inference_deployment import get_deployment_info_by_api_key
from app.inference_model import get_model_info_by_api_key_and_model_name
from app.inference_usage import check_api_key_usage
from app.openai_client import get_client
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

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
        body.temperature = 0
        body.stream = True
        body.stream_options = StreamOptions(include_usage=True)
        body_data = {
            key: value for key, value in body.model_dump().items() if value is not None
        }
        print("body_data", body_data)
        response = client.chat.completions.create(**body_data)

        return StreamingResponse(
            generate_response(
                request, response, encoder, total_tokens, model_id, api_key_id
            ),
            media_type="text/event-stream",
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


async def generate_response(
    request, response, encoder, total_tokens, model_id, api_key_id
):
    try:
        for chunk in response:
            if chunk.usage is not None:
                total_tokens["prompt"] = chunk.usage.prompt_tokens
                total_tokens["completion"] = chunk.usage.completion_tokens
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

                token_usage_queue = request.app.state.token_usage_queue
                await token_usage_queue.put(
                    (
                        model_id,
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
