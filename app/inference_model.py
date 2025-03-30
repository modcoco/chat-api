from datetime import datetime
from typing import Optional, Tuple

import asyncpg
from models import InferenceModelCreate
from fastapi import HTTPException


async def create_inference_model(conn, model: InferenceModelCreate):
    # 1. 检查部署是否存在且可用
    query_check_deployment = """
    SELECT id, status, is_deleted 
    FROM inference_deployment 
    WHERE id = $1
    """
    deployment = await conn.fetchrow(query_check_deployment, model.inference_id)

    if not deployment:
        raise HTTPException(status_code=404, detail="Inference deployment not found.")

    if deployment["is_deleted"] or deployment["status"] != "active":
        raise HTTPException(
            status_code=400,
            detail="Inference deployment is either deleted or not in active status.",
        )

    # 2. 检查模型是否已注册（inference_id + model_id 唯一）
    query_check_existing = """
    SELECT id 
    FROM inference_model 
    WHERE inference_id = $1 AND model_id = $2 AND is_deleted = FALSE
    """
    existing_model = await conn.fetchrow(
        query_check_existing, model.inference_id, model.model_id
    )

    if existing_model:
        raise HTTPException(
            status_code=400,
            detail="Model already registered with this inference deployment.",
        )

    # 3. 插入新模型
    query = """
    INSERT INTO inference_model (
        model_name, visibility, inference_id, 
        model_id, max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, 
        created_at, updated_at
    ) 
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) 
    RETURNING id, model_name, visibility, inference_id, model_id, 
              max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, created_at, updated_at;
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
        current_time,  # created_at 字段
        current_time,  # updated_at 字段
    )

    result_dict = dict(result)
    result_dict["created_at"] = result_dict["created_at"].isoformat()
    result_dict["updated_at"] = result_dict["updated_at"].isoformat()

    result_dict = {k: v for k, v in result_dict.items() if v is not None}

    return result_dict


async def get_model_info_by_api_key_and_model_name(
    api_key: str, model_name: str, db: asyncpg.Pool
) -> Optional[Tuple[int, str]]:
    """
    根据API Key和模型名称获取模型信息

    参数:
        api_key: 用户提供的API Key
        model_name: 请求的模型名称
        db: 数据库连接池

    返回:
        元组 (model_id: int, model_path: str) 或 None
    """
    query = """
    SELECT im.id AS model_id, im.model_id AS model_path
    FROM inference_api_key iak
    JOIN inference_api_key_model iakm ON iak.id = iakm.api_key_id
    JOIN inference_model im ON iakm.model_id = im.id
    WHERE iak.api_key = $1
    AND im.model_name = $2
    AND iak.is_deleted = FALSE
    AND im.is_deleted = FALSE
    """

    async with db.acquire() as conn:
        result = await conn.fetchrow(query, api_key, model_name)

    if result:
        return result["model_id"], result["model_path"]  # 返回 (int, str)
    return None
