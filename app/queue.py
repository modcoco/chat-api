from fastapi import FastAPI

from app.inference_usage import insert_token_usage


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
