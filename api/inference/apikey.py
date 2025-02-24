from datetime import datetime, timedelta
from typing import List
import uuid
from fastapi import APIRouter, HTTPException, Request
from grpc import Status

from models import InferenceModelApiKeyCreate, InferenceModelApiKeyResponse


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
