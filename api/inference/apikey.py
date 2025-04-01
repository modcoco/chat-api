from datetime import datetime, timedelta
from typing import List, Optional
import uuid
from fastapi import APIRouter, HTTPException, Request
from fastapi import status

from models import (
    MultiModelApiKeyCreate,
    MultiModelApiKeyResponse,
    QuotaUpdateRequest,
    QuotaUpdateResponse,
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

    # Calculate expiration time
    expires_at = None
    if api_key_data.active_days:
        expires_at = datetime.now() + timedelta(days=api_key_data.active_days)

    created_at_time = datetime.now()

    async with db.acquire() as conn:
        # Start transaction
        async with conn.transaction():
            # 1. Create main API Key record
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

            # 2. Create quota records for each model
            key_id = key_result["id"]
            models = []

            for model_quota in api_key_data.model_quotas:
                # Check if model exists and is valid
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

                # Insert model quota record
                query_insert_quota = """
                INSERT INTO inference_api_key_model (
                    api_key_id, model_id, 
                    max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota,
                    created_at
                ) 
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, model_id, max_token_quota, max_prompt_tokens_quota, 
                          max_completion_tokens_quota, created_at, is_deleted;
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

                models.append(
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
                        "is_deleted": quota_result["is_deleted"],
                    }
                )

    # Build response
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
        "models": models,  # Changed from model_quotas to models
    }



@router.patch("/api-key-model/{relation_id}/quotas", response_model=QuotaUpdateResponse)
async def update_api_key_model_quotas(
    request: Request,
    relation_id: int,
    quota_data: QuotaUpdateRequest
):
    """
    Update quota limits for an API key-model relation
    
    Parameters:
    - relation_id: The ID from inference_api_key_model table
    - quota_data: JSON body containing any of these optional fields:
        - max_token_quota
        - max_prompt_tokens_quota  
        - max_completion_tokens_quota
    
    At least one quota field must be provided.
    """
    # Convert model to dict and remove unset fields
    update_data = quota_data.dict(exclude_unset=True)
    
    # Validate at least one field was provided
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one quota field must be provided for update"
        )

    # Build dynamic update query
    updates = []
    params = []
    
    # Add fields that were provided
    if 'max_token_quota' in update_data:
        updates.append("max_token_quota = $1")
        params.append(update_data['max_token_quota'])
    
    if 'max_prompt_tokens_quota' in update_data:
        position = len(params) + 1
        updates.append(f"max_prompt_tokens_quota = ${position}")
        params.append(update_data['max_prompt_tokens_quota'])
    
    if 'max_completion_tokens_quota' in update_data:
        position = len(params) + 1
        updates.append(f"max_completion_tokens_quota = ${position}")
        params.append(update_data['max_completion_tokens_quota'])
    
    # Add timestamp update
    updates.append("updated_at = NOW()")
    
    # Build final query
    query = f"""
        UPDATE inference_api_key_model
        SET {', '.join(updates)}
        WHERE id = ${len(params) + 1} AND is_deleted = FALSE
        RETURNING 
            id,
            api_key_id,
            model_id,
            max_token_quota,
            max_prompt_tokens_quota,
            max_completion_tokens_quota,
            updated_at;
    """
    params.append(relation_id)

    # Execute query
    db = request.app.state.db_pool
    async with db.acquire() as conn:
        result = await conn.fetchrow(query, *params)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key-model relation not found or already deleted"
            )

    return {
        "id": result["id"],
        "api_key_id": result["api_key_id"],
        "model_id": result["model_id"],
        "max_token_quota": result["max_token_quota"],
        "max_prompt_tokens_quota": result["max_prompt_tokens_quota"],
        "max_completion_tokens_quota": result["max_completion_tokens_quota"],
        "updated_at": result["updated_at"],
    }

@router.get("/apikey", response_model=List[MultiModelApiKeyResponse])
async def get_multi_model_api_keys(request: Request):
    async with request.app.state.db_pool.acquire() as conn:
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
                iakm.id AS relation_id,  -- Add this line to get the relation ID
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
            AND iakm.is_deleted = FALSE  -- Added this condition
            AND im.is_deleted = FALSE
        )
        SELECT 
            k.*,
            q.relation_id,  -- Include relation_id in the SELECT
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

            if record["quota_model_id"]:
                response_map[key_id]["models"].append(
                    {
                        "relation_id": record["relation_id"],  # Add relation_id here
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
    db = request.app.state.db_pool
    async with db.acquire() as conn:
        # Start a transaction
        async with conn.transaction():
            # Check if API key exists and is not already deleted
            result = await conn.fetchrow(
                "SELECT id FROM inference_api_key WHERE id = $1 AND is_deleted = FALSE;",
                api_key_id,
            )
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="API key not found or already deleted",
                )

            # Mark all associated models as deleted first
            await conn.execute(
                """
                UPDATE inference_api_key_model
                SET is_deleted = TRUE, deleted_at = NOW()
                WHERE api_key_id = $1 AND is_deleted = FALSE;
                """,
                api_key_id,
            )

            # Then mark the API key as deleted
            updated_result = await conn.fetchrow(
                """
                UPDATE inference_api_key
                SET is_deleted = TRUE, deleted_at = NOW()
                WHERE id = $1
                RETURNING id, is_deleted;
                """,
                api_key_id,
            )

    return {"id": updated_result["id"], "isDeleted": updated_result["is_deleted"]}
