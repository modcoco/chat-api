import time
import json
from flask import Flask, request, Response
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
import tiktoken

app = Flask(__name__)

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

encoder = tiktoken.get_encoding("cl100k_base")


@app.route("/v1/chat/completions", methods=["POST"])
def proxy_openai():
    try:
        user_request = request.get_json()

        total_prompt_tokens = 0
        total_completion_tokens = 0

        messages = user_request.get("messages", [])
        prompt_text = " ".join([msg.get("content", "") for msg in messages])
        total_prompt_tokens = len(encoder.encode(prompt_text))

        response = client.chat.completions.create(
            model="/mnt/data/models/deepseek-ai_DeepSeek-R1",
            messages=messages,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )

        def generate_response():
            nonlocal total_prompt_tokens, total_completion_tokens

            try:
                for chunk in response:
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
            except GeneratorExit:
                print("用户主动中断了流式请求")
                print(f"Total Prompt Tokens (手动计算): {total_prompt_tokens}")
                print(f"Total Completion Tokens (手动计算): {total_completion_tokens}")
                raise
            return

        return Response(generate_response(), content_type="text/event-stream")

    except Exception as e:
        print(f"Error: {e}")
        return Response("Internal Server Error", status=500)


if __name__ == "__main__":
    app.run(port=5000)


