from datetime import datetime, timedelta
from typing import List
import uuid
from fastapi import APIRouter, HTTPException, Request
from fastapi import status

from models import (
    MultiModelApiKeyCreate,
    MultiModelApiKeyResponse,
)


router = APIRouter()


@router.post("/apikey", response_model=MultiModelApiKeyResponse)
async def create_multi_model_api_key(
    request: Request,
    api_key_data: MultiModelApiKeyCreate,
):
    db = request.app.state.db_pool
    unique_id = uuid.uuid4().hex
    api_key = f"sk-{unique_id}"

    # 计算过期时间
    expires_at = None
    if api_key_data.active_days:
        expires_at = datetime.now() + timedelta(days=api_key_data.active_days)

    created_at_time = datetime.now()

    async with db.acquire() as conn:
        # 开始事务
        async with conn.transaction():
            # 1. 创建主API Key记录
            query_insert_key = """
            INSERT INTO inference_api_key (
                api_key_name, api_key, active_days, created_at, expires_at
            ) 
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, api_key_name, api_key, active_days, 
                      created_at, expires_at, is_deleted;
            """
            key_result = await conn.fetchrow(
                query_insert_key,
                api_key_data.api_key_name,
                api_key,
                api_key_data.active_days,
                created_at_time,
                expires_at,
            )

            # 2. 为每个模型创建配额记录
            key_id = key_result["id"]
            model_responses = []

            for model_quota in api_key_data.model_quotas:
                # 检查模型是否存在且有效
                query_check_model = """
                SELECT im.id, im.model_name, idp.is_deleted as idp_is_deleted, idp.status
                FROM inference_model im
                JOIN inference_deployment idp ON im.inference_id = idp.id
                WHERE im.id = $1 AND im.is_deleted = FALSE
                """
                model_result = await conn.fetchrow(
                    query_check_model, model_quota.model_id
                )

                if not model_result:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Model with id {model_quota.model_id} not found or marked as deleted",
                    )

                if model_result["idp_is_deleted"]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"The associated inference deployment for model {model_quota.model_id} is marked as deleted",
                    )

                if model_result["status"] != "active":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"The associated inference deployment for model {model_quota.model_id} is not active",
                    )

                # 插入模型配额记录
                query_insert_quota = """
                INSERT INTO inference_api_key_model (
                    api_key_id, model_id, 
                    max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota,
                    created_at
                ) 
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, model_id, max_token_quota, max_prompt_tokens_quota, 
                          max_completion_tokens_quota, created_at;
                """
                quota_result = await conn.fetchrow(
                    query_insert_quota,
                    key_id,
                    model_quota.model_id,
                    model_quota.max_token_quota,
                    model_quota.max_prompt_tokens_quota,
                    model_quota.max_completion_tokens_quota,
                    created_at_time,
                )

                model_responses.append(
                    {
                        "model_id": quota_result["model_id"],
                        "model_name": model_result["model_name"],
                        "max_token_quota": quota_result["max_token_quota"],
                        "max_prompt_tokens_quota": quota_result[
                            "max_prompt_tokens_quota"
                        ],
                        "max_completion_tokens_quota": quota_result[
                            "max_completion_tokens_quota"
                        ],
                        "created_at": quota_result["created_at"].isoformat(),
                    }
                )

    # 构建响应
    return {
        "id": key_result["id"],
        "api_key_name": key_result["api_key_name"],
        "api_key": key_result["api_key"],
        "active_days": key_result["active_days"],
        "created_at": key_result["created_at"].isoformat(),
        "expires_at": (
            key_result["expires_at"].isoformat() if key_result["expires_at"] else None
        ),
        "is_deleted": key_result["is_deleted"],
        "model_quotas": model_responses,
    }


@router.get("/apikey", response_model=List[MultiModelApiKeyResponse])
async def get_multi_model_api_keys(request: Request):
    async with request.app.state.db_pool.acquire() as conn:
        # 使用COALESCE确保NULL值转为0
        query = """
        WITH api_keys AS (
            SELECT id, api_key_name, api_key, active_days, 
                   created_at, last_used_at, expires_at
            FROM inference_api_key
            WHERE is_deleted = FALSE
            ORDER BY created_at DESC
        ),
        quotas AS (
            SELECT 
                iakm.api_key_id, 
                iakm.model_id, 
                im.model_name,
                iakm.max_token_quota, 
                iakm.max_prompt_tokens_quota, 
                iakm.max_completion_tokens_quota, 
                iakm.created_at
            FROM inference_api_key_model iakm
            JOIN inference_model im ON iakm.model_id = im.id
            WHERE iakm.api_key_id IN (SELECT id FROM api_keys)
            AND im.is_deleted = FALSE
        )
        SELECT 
            k.*,
            q.model_id AS quota_model_id,
            q.model_name,
            q.max_token_quota,
            q.max_prompt_tokens_quota,
            q.max_completion_tokens_quota,
            q.created_at AS quota_created_at,
            COALESCE((
                SELECT SUM(prompt_tokens) 
                FROM inference_api_key_token_usage 
                WHERE api_key_id = k.id AND model_id = q.model_id
            ), 0) AS used_prompt_tokens,
            COALESCE((
                SELECT SUM(completion_tokens) 
                FROM inference_api_key_token_usage 
                WHERE api_key_id = k.id AND model_id = q.model_id
            ), 0) AS used_completion_tokens,
            COALESCE((
                SELECT SUM(prompt_tokens + completion_tokens) 
                FROM inference_api_key_token_usage 
                WHERE api_key_id = k.id AND model_id = q.model_id
            ), 0) AS used_total_tokens
        FROM api_keys k
        LEFT JOIN quotas q ON k.id = q.api_key_id
        ORDER BY k.created_at DESC, q.created_at;
        """

        records = await conn.fetch(query)

        # 按API Key分组数据
        response_map = {}
        for record in records:
            key_id = record["id"]

            if key_id not in response_map:
                response_map[key_id] = {
                    "id": key_id,
                    "api_key_name": record["api_key_name"],
                    "api_key": record["api_key"],
                    "active_days": record["active_days"],
                    "created_at": record["created_at"].isoformat(),
                    "last_used_at": (
                        record["last_used_at"].isoformat()
                        if record["last_used_at"]
                        else None
                    ),
                    "expires_at": (
                        record["expires_at"].isoformat()
                        if record["expires_at"]
                        else None
                    ),
                    "models": [],
                }

            if record["quota_model_id"]:  # 只添加有模型配额的数据
                response_map[key_id]["models"].append(
                    {
                        "model_id": record["quota_model_id"],
                        "model_name": record["model_name"],
                        "max_token_quota": record["max_token_quota"],
                        "max_prompt_tokens_quota": record["max_prompt_tokens_quota"],
                        "max_completion_tokens_quota": record[
                            "max_completion_tokens_quota"
                        ],
                        "used_prompt_tokens": record["used_prompt_tokens"],
                        "used_completion_tokens": record["used_completion_tokens"],
                        "used_total_tokens": record["used_total_tokens"],
                        "created_at": record["quota_created_at"].isoformat(),
                    }
                )

    return list(response_map.values())


@router.patch("/apikey/{api_key_id}/delete")
async def delete_multi_model_api_key(
    request: Request,
    api_key_id: int,
):
    query_check = """
    SELECT id FROM inference_api_key 
    WHERE id = $1 AND is_deleted = FALSE;
    """

    update_query = """
    UPDATE inference_api_key
    SET is_deleted = TRUE, deleted_at = NOW()
    WHERE id = $1
    RETURNING id, is_deleted;
    """

    db = request.app.state.db_pool
    async with db.acquire() as conn:
        result = await conn.fetchrow(query_check, api_key_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found or already deleted",
            )

        updated_result = await conn.fetchrow(update_query, api_key_id)

    return {"id": updated_result["id"], "isDeleted": updated_result["is_deleted"]}
