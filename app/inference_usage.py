# check_model_quota 是否检查模型配哦额
import asyncio
from datetime import datetime
import time
from typing import Optional
import asyncpg


async def check_api_key_usage(
    model_id: int, api_key: str, db: asyncpg.Pool, check_model_quota: bool = True
) -> Optional[str]:
    # 首先检查 API Key 是否存在且有效
    api_key_query = """
    SELECT id, expires_at, is_deleted
    FROM inference_api_key
    WHERE api_key = $1;
    """

    async with db.acquire() as conn:
        api_key_result = await conn.fetchrow(api_key_query, api_key)

    if not api_key_result:
        return "API Key not found"

    if api_key_result["is_deleted"]:
        return "API Key is deleted"

    if api_key_result["expires_at"] and api_key_result["expires_at"] < datetime.now():
        return "API Key has expired"

    api_key_id = api_key_result["id"]

    # 获取该 API Key 对指定模型的配额信
    model_quota_query = """
    SELECT max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota
    FROM inference_api_key_model
    WHERE api_key_id = $1 AND model_id = $2;
    """

    async with db.acquire() as conn:
        model_quota = await conn.fetchrow(model_quota_query, api_key_id, model_id)

    if not model_quota:
        return "API Key does not have access to this model"

    max_token_quota = model_quota["max_token_quota"]
    max_prompt_tokens_quota = model_quota["max_prompt_tokens_quota"]
    max_completion_tokens_quota = model_quota["max_completion_tokens_quota"]

    # 获取该 api_key 对指定模型的所有 token 使用记录
    usage_query = """
    SELECT SUM(prompt_tokens) AS total_prompt_tokens, SUM(completion_tokens) AS total_completion_tokens
    FROM inference_api_key_token_usage
    WHERE api_key_id = $1 AND model_id = $2;
    """

    async with db.acquire() as conn:
        usage = await conn.fetchrow(usage_query, api_key_id, model_id)

    total_prompt_tokens = usage["total_prompt_tokens"] or 0 if usage else 0
    total_completion_tokens = usage["total_completion_tokens"] or 0 if usage else 0
    total_tokens = total_prompt_tokens + total_completion_tokens

    # 打印 API Key 的配额使用情况
    print("\nAPI Key Quota Usage:")
    print(f"Model ID: {model_id}")
    print(f"API Key ID: {api_key_id}")

    if max_prompt_tokens_quota is not None:
        remaining_prompt = max_prompt_tokens_quota - total_prompt_tokens
        print(
            f"Prompt Tokens - Quota: {max_prompt_tokens_quota}, Used: {total_prompt_tokens}, Remaining: {remaining_prompt}"
        )
    else:
        print("Prompt Tokens - No quota limit")

    if max_completion_tokens_quota is not None:
        remaining_completion = max_completion_tokens_quota - total_completion_tokens
        print(
            f"Completion Tokens - Quota: {max_completion_tokens_quota}, Used: {total_completion_tokens}, Remaining: {remaining_completion}"
        )
    else:
        print("Completion Tokens - No quota limit")

    if max_token_quota is not None:
        remaining_total = max_token_quota - total_tokens
        print(
            f"Total Tokens - Quota: {max_token_quota}, Used: {total_tokens}, Remaining: {remaining_total}"
        )
    else:
        print("Total Tokens - No quota limit")

    # 判断是否超出 API Key 对该模型的配额
    if (
        max_prompt_tokens_quota is not None
        and total_prompt_tokens > max_prompt_tokens_quota
    ):
        return "Prompt tokens exceeded quota for this model"

    if (
        max_completion_tokens_quota is not None
        and total_completion_tokens > max_completion_tokens_quota
    ):
        return "Completion tokens exceeded quota for this model"

    if max_token_quota is not None and total_tokens > max_token_quota:
        return "Total tokens exceeded quota for this model"

    # 如果 API key 的配额没有超出，则进行模型全局配额检查
    if check_model_quota:
        model_check_res = await check_model_usage(model_id, db)
        if model_check_res:
            return model_check_res

    # 如果没有超出配额
    return None


async def check_model_usage(model_id: int, db: asyncpg.Pool) -> Optional[str]:
    # 获取模型的全局配额信息
    model_query = """
    SELECT max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota
    FROM inference_model
    WHERE id = $1;
    """

    async with db.acquire() as conn:
        model = await conn.fetchrow(model_query, model_id)

    if not model:
        return "Model not found"

    model_max_token_quota = model["max_token_quota"]
    model_max_prompt_tokens_quota = model["max_prompt_tokens_quota"]
    model_max_completion_tokens_quota = model["max_completion_tokens_quota"]

    # 如果模型没有设置配额限制，直接返回
    if all(
        q is None
        for q in [
            model_max_token_quota,
            model_max_prompt_tokens_quota,
            model_max_completion_tokens_quota,
        ]
    ):
        print("\nModel Quota: No quota limits set")
        return None

    # 获取该模型的全局使用量
    model_usage_query = """
    SELECT SUM(prompt_tokens) AS total_prompt_tokens, SUM(completion_tokens) AS total_completion_tokens
    FROM inference_api_key_token_usage
    WHERE model_id = $1;
    """

    async with db.acquire() as conn:
        model_usage = await conn.fetchrow(model_usage_query, model_id)

    total_model_prompt_tokens = (
        model_usage["total_prompt_tokens"] or 0 if model_usage else 0
    )
    total_model_completion_tokens = (
        model_usage["total_completion_tokens"] or 0 if model_usage else 0
    )
    total_model_tokens = total_model_prompt_tokens + total_model_completion_tokens

    # 打印模型的全局配额使用情况
    print("\nModel Global Quota Usage:")
    print(f"Model ID: {model_id}")

    if model_max_prompt_tokens_quota is not None:
        remaining_prompt = model_max_prompt_tokens_quota - total_model_prompt_tokens
        print(
            f"Global Prompt Tokens - Quota: {model_max_prompt_tokens_quota}, Used: {total_model_prompt_tokens}, Remaining: {remaining_prompt}"
        )
    else:
        print("Global Prompt Tokens - No quota limit")

    if model_max_completion_tokens_quota is not None:
        remaining_completion = (
            model_max_completion_tokens_quota - total_model_completion_tokens
        )
        print(
            f"Global Completion Tokens - Quota: {model_max_completion_tokens_quota}, Used: {total_model_completion_tokens}, Remaining: {remaining_completion}"
        )
    else:
        print("Global Completion Tokens - No quota limit")

    if model_max_token_quota is not None:
        remaining_total = model_max_token_quota - total_model_tokens
        print(
            f"Global Total Tokens - Quota: {model_max_token_quota}, Used: {total_model_tokens}, Remaining: {remaining_total}"
        )
    else:
        print("Global Total Tokens - No quota limit")

    # 检查模型的全局配额是否超出
    if (
        model_max_prompt_tokens_quota is not None
        and total_model_prompt_tokens > model_max_prompt_tokens_quota
    ):
        return "Model's global prompt tokens exceeded quota"

    if (
        model_max_completion_tokens_quota is not None
        and total_model_completion_tokens > model_max_completion_tokens_quota
    ):
        return "Model's global completion tokens exceeded quota"

    if model_max_token_quota is not None and total_model_tokens > model_max_token_quota:
        return "Model's global total tokens exceeded quota"

    return None


async def insert_token_usage(
    db_pool,
    model_id,
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
                        INSERT INTO inference_api_key_token_usage
                        (model_id, completions_chunk_id, api_key_id, prompt_tokens, completion_tokens, type, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        model_id,
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
