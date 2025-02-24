from datetime import datetime, timedelta
from typing import List
import uuid
from fastapi import APIRouter, HTTPException, Request
from grpc import Status

from models import InferenceModelApiKeyCreate, InferenceModelApiKeyResponse


router = APIRouter()


@router.post("/api-key", response_model=dict)
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

    created_time = datetime.now()

    query = """
    INSERT INTO inference_model_api_key (
        api_key_name, inference_model_id, api_key, 
        max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, 
        created, expires_at,active_days
    ) 
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) 
    RETURNING id, api_key_name, inference_model_id, api_key, 
              max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, 
              created, expires_at, active_days, is_deleted;
    """

    async with db.acquire() as conn:
        result = await conn.fetchrow(
            query,
            api_key_data.api_key_name,
            api_key_data.inference_model_id,
            api_key,
            api_key_data.max_token_quota,
            api_key_data.max_prompt_tokens_quota,
            api_key_data.max_completion_tokens_quota,
            created_time,
            expires_at,
            api_key_data.active_days,
        )

    return dict(result)


@router.get("/api-keys", response_model=List[InferenceModelApiKeyResponse])
async def get_inference_model_api_keys(
    request: Request,
):
    query = """
    SELECT id, api_key_name, inference_model_id, api_key, 
           max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota,
           active_days, created, last_used_at, expires_at, is_deleted
    FROM inference_model_api_key
    WHERE is_deleted = FALSE;
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
            "created": row["created"].isoformat(),
            "last_used_at": (
                row["last_used_at"].isoformat() if row["last_used_at"] else None
            ),
            "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
            "is_deleted": row["is_deleted"],
        }
        for row in result
    ]


@router.patch("/api-key/{api_key_id}/delete")
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
            status_code=Status.HTTP_404_NOT_FOUND,
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
