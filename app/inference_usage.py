# check_model_quota 是否检查模型配哦额
import asyncio
from datetime import datetime
import time
from typing import Optional
import asyncpg


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
    AND created_at <= $2;  -- 可选：限制到某个时间范围（例如当前时间前）
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
    AND created_at <= $2;
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
                        (completions_chunk_id, api_key_id, prompt_tokens, completion_tokens, type, created_at)
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
