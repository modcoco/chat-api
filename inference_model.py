from datetime import datetime
from typing import Optional

import asyncpg
from models import InferenceModelCreate


async def create_inference_model(conn, model: InferenceModelCreate):
    query = """
    INSERT INTO inference_model (
        model_name, visibility, inference_id, 
        model_id, max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, 
        created, updated
    ) 
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) 
    RETURNING id, model_name, visibility, inference_id, model_id, 
              max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, created, updated;
    """

    current_time = datetime.now()

    result = await conn.fetchrow(
        query,
        model.model_name,
        model.visibility,
        model.inference_id,
        model.model_id,
        model.max_token_quota,
        model.max_prompt_tokens_quota,
        model.max_completion_tokens_quota,
        current_time,  # created 字段
        current_time,  # updated 字段
    )

    result_dict = dict(result)
    result_dict["created"] = result_dict["created"].isoformat()
    result_dict["updated"] = result_dict["updated"].isoformat()

    result_dict = {k: v for k, v in result_dict.items() if v is not None}

    return result_dict


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
