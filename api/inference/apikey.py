from datetime import datetime, timedelta
from typing import List
import uuid
from fastapi import APIRouter, HTTPException, Request
from fastapi import status 

from models import (
    InferenceModelApiKeyCreate,
    InferenceModelApiKeyResponse,
    MultiModelApiKeyCreate,
    MultiModelApiKeyResponse,
)


router = APIRouter()


@router.post("/apikey", response_model=dict)
async def create_inference_model_api_key(
    request: Request,
    api_key_data: InferenceModelApiKeyCreate,
):
    db = request.app.state.db_pool
    unique_id = uuid.uuid4().hex
    api_key = f"sk-{unique_id}"

    if api_key_data.active_days:
        expires_at = datetime.now() + timedelta(days=api_key_data.active_days)
    else:
        expires_at = None

    created_at_time = datetime.now()

    # Check if the corresponding inference_model exists and is not deleted
    query_check_model = """
    SELECT im.id, idp.is_deleted as idp_is_deleted, idp.status
    FROM inference_model im
    JOIN inference_deployment idp ON im.inference_id = idp.id
    WHERE im.id = $1 AND im.is_deleted = FALSE
    """

    async with db.acquire() as conn:
        # Check model existence and validity
        result = await conn.fetchrow(query_check_model, api_key_data.inference_model_id)

        if not result:
            raise HTTPException(
                status_code=404,
                detail="Inference model not found or it is marked as deleted",
            )

        if result["idp_is_deleted"]:
            raise HTTPException(
                status_code=400,
                detail="The associated inference deployment is marked as deleted",
            )

        if result["status"] != "active":
            raise HTTPException(
                status_code=400,
                detail="The associated inference deployment is not active",
            )

    # Proceed with inserting the API key
    query_insert = """
    INSERT INTO inference_model_api_key (
        api_key_name, inference_model_id, api_key, 
        max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, 
        created_at, expires_at, active_days
    ) 
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) 
    RETURNING id, api_key_name, inference_model_id, api_key, 
              max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, 
              created_at, expires_at, active_days, is_deleted;
    """

    async with db.acquire() as conn:
        result = await conn.fetchrow(
            query_insert,
            api_key_data.api_key_name,
            api_key_data.inference_model_id,
            api_key,
            api_key_data.max_token_quota,
            api_key_data.max_prompt_tokens_quota,
            api_key_data.max_completion_tokens_quota,
            created_at_time,
            expires_at,
            api_key_data.active_days,
        )

    return dict(result)


@router.get("/apikey", response_model=List[InferenceModelApiKeyResponse])
async def get_inference_model_api_keys(
    request: Request,
):
    query = """
    SELECT imak.id, imak.api_key_name, imak.inference_model_id, imak.api_key, 
           imak.max_token_quota, imak.max_prompt_tokens_quota, imak.max_completion_tokens_quota,
           imak.active_days, imak.created_at, imak.last_used_at, imak.expires_at, imak.is_deleted
    FROM inference_model_api_key imak
    JOIN inference_model im ON im.id = imak.inference_model_id
    JOIN inference_deployment idp ON im.inference_id = idp.id
    WHERE imak.is_deleted = FALSE
    AND im.is_deleted = FALSE
    AND idp.is_deleted = FALSE
    AND idp.status = 'active';
    """

    db = request.app.state.db_pool
    async with db.acquire() as conn:
        result = await conn.fetch(query)

    return [
        {
            "id": row["id"],
            "api_key_name": row["api_key_name"],
            "inference_model_id": row["inference_model_id"],
            "api_key": row["api_key"],
            "max_token_quota": row.get("max_token_quota"),
            "max_prompt_tokens_quota": row.get("max_prompt_tokens_quota"),
            "max_completion_tokens_quota": row.get("max_completion_tokens_quota"),
            "active_days": row.get("active_days"),
            "created_at": row["created_at"].isoformat(),
            "last_used_at": (
                row["last_used_at"].isoformat() if row["last_used_at"] else None
            ),
            "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
            "is_deleted": row["is_deleted"],
        }
        for row in result
    ]


@router.patch("/apikey/{api_key_id}/delete")
async def delete_inference_model_api_key(
    request: Request,
    api_key_id: int,
):
    query = """
    SELECT id FROM inference_model_api_key WHERE id = $1 AND is_deleted = FALSE;
    """
    db = request.app.state.db_pool
    async with db.acquire() as conn:
        result = await conn.fetchrow(query, api_key_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or already deleted",
        )

    update_query = """
    UPDATE inference_model_api_key
    SET is_deleted = TRUE
    WHERE id = $1
    RETURNING id, is_deleted;
    """

    async with db.acquire() as conn:
        updated_result = await conn.fetchrow(update_query, api_key_id)

    return {"id": updated_result["id"], "is_deleted": updated_result["is_deleted"]}


@router.post("/multimodel/apikey", response_model=MultiModelApiKeyResponse)
async def create_multi_model_api_key(
    request: Request,
    api_key_data: MultiModelApiKeyCreate,
):
    db = request.app.state.db_pool
    unique_id = uuid.uuid4().hex
    api_key = f"sk-multi-{unique_id}"

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


@router.get("/multimodel/apikey", response_model=List[MultiModelApiKeyResponse])
async def get_multi_model_api_keys(
    request: Request,
):
    # 查询主API Key信息
    query_keys = """
    SELECT id, api_key_name, api_key, active_days, 
           created_at, last_used_at, expires_at, is_deleted
    FROM inference_api_key
    WHERE is_deleted = FALSE
    ORDER BY created_at DESC;
    """

    # 查询每个API Key的模型配额信息
    query_quotas = """
    SELECT iakm.api_key_id, iakm.model_id, im.model_name,
           iakm.max_token_quota, iakm.max_prompt_tokens_quota, 
           iakm.max_completion_tokens_quota, iakm.created_at
    FROM inference_api_key_model iakm
    JOIN inference_model im ON iakm.model_id = im.id
    WHERE iakm.api_key_id = ANY($1::int[])
    AND im.is_deleted = FALSE
    ORDER BY iakm.created_at;
    """

    db = request.app.state.db_pool
    async with db.acquire() as conn:
        keys = await conn.fetch(query_keys)
        if not keys:
            return []

        # 获取这些Key的所有模型配额
        key_ids = [key["id"] for key in keys]
        quotas = await conn.fetch(query_quotas, key_ids)

        # 按API Key ID分组配额
        quotas_by_key = {}
        for quota in quotas:
            key_id = quota["api_key_id"]
            if key_id not in quotas_by_key:
                quotas_by_key[key_id] = []
            quotas_by_key[key_id].append(quota)

    # 构建响应
    response = []
    for key in keys:
        key_id = key["id"]
        model_quotas = quotas_by_key.get(key_id, [])

        response.append(
            {
                "id": key["id"],
                "api_key_name": key["api_key_name"],
                "api_key": key["api_key"],
                "active_days": key["active_days"],
                "created_at": key["created_at"].isoformat(),
                "last_used_at": (
                    key["last_used_at"].isoformat() if key["last_used_at"] else None
                ),
                "expires_at": (
                    key["expires_at"].isoformat() if key["expires_at"] else None
                ),
                "is_deleted": key["is_deleted"],
                "model_quotas": [
                    {
                        "model_id": q["model_id"],
                        "model_name": q["model_name"],
                        "max_token_quota": q["max_token_quota"],
                        "max_prompt_tokens_quota": q["max_prompt_tokens_quota"],
                        "max_completion_tokens_quota": q["max_completion_tokens_quota"],
                        "created_at": q["created_at"].isoformat(),
                    }
                    for q in model_quotas
                ],
            }
        )

    return response


@router.patch("/multimodel/apikey/{api_key_id}/delete")
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
    
    return {"id": updated_result["id"], "is_deleted": updated_result["is_deleted"]}