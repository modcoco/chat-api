from contextlib import asynccontextmanager
import os
import time
import json
import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
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
    yield
    await pool.close()
    print("Closing database connection pool...")


app = FastAPI(lifespan=lifespan)


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

encoder = tiktoken.get_encoding("cl100k_base")


@app.post("/v1/chat/completions")
async def proxy_openai(request: Request):
    try:
        # 解析请求体
        user_request = await request.json()

        total_prompt_tokens = 0
        total_completion_tokens = 0

        # 计算 prompt tokens
        messages = user_request.get("messages", [])
        prompt_text = " ".join([msg.get("content", "") for msg in messages])
        total_prompt_tokens = len(encoder.encode(prompt_text))

        # 调用 OpenAI 客户端，获取流式响应
        response = client.chat.completions.create(
            model="/mnt/data/models/deepseek-ai_DeepSeek-R1",
            messages=messages,
            temperature=0,
            stream=True,  # 确保 stream=True
            stream_options={"include_usage": True},
        )

        # 生成流式响应
        async def generate_response():
            nonlocal total_prompt_tokens, total_completion_tokens

            try:
                for chunk in response:  # 使用同步 for 处理流式响应
                    if chunk.usage is not None:
                        # 如果能拿到 chunk.usage，直接使用它的值
                        total_prompt_tokens = chunk.usage.prompt_tokens
                        total_completion_tokens = chunk.usage.completion_tokens
                        print(f"Total Prompt Tokens: {total_prompt_tokens}")
                        print(f"Total Completion Tokens: {total_completion_tokens}")
                        break

                    # 如果没有 chunk.usage，手动计算 completion_tokens
                    for choice in chunk.choices:
                        if choice.delta.content:
                            total_completion_tokens += len(
                                encoder.encode(choice.delta.content)
                            )

                    # 检查客户端是否断开连接
                    if await request.is_disconnected():  # 使用 await 调用异步方法
                        print("客户端主动断开连接")
                        print(f"Total Prompt Tokens (手动计算): {total_prompt_tokens}")
                        print(f"Total Completion Tokens (手动计算): {total_completion_tokens}")
                        return

                    # 构建流式响应块
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
        return StreamingResponse(generate_response(), media_type="text/event-stream")

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/v1/models")
async def get_users():
    async with app.state.db_pool.acquire() as connection:
        result = await connection.fetch("SELECT * FROM models")
        return {"users": [dict(record) for record in result]}  # 将结果转换为字典


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
