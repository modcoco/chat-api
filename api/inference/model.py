from typing import List
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.inference_model import create_inference_model
from models import InferenceModelCreate


router = APIRouter()


@router.post("/model", response_model=InferenceModelCreate)
async def add_inference_model(
    request: Request,
    model: InferenceModelCreate,
):
    db = request.app.state.db_pool
    async with db.acquire() as conn:
        inserted_model = await create_inference_model(conn, model)

        camel_case_model = {
            snake_to_camel(key): value for key, value in inserted_model.items()
        }

        return JSONResponse(content=camel_case_model, status_code=201)


def snake_to_camel(snake_str: str) -> str:
    """将 snake_case 转换为 camelCase"""
    parts = snake_str.split("_")
    return parts[0] + "".join(x.capitalize() for x in parts[1:])


@router.get("/model", response_model=List[dict])
async def get_all_inference_services(
    request: Request,
):
    db = request.app.state.db_pool
    query = """
    SELECT 
        im.id,
        im.model_name,
        im.visibility,
        im.inference_id,
        im.model_id,
        im.max_token_quota,
        im.max_prompt_tokens_quota,
        im.max_completion_tokens_quota,
        im.created_at,
        im.updated_at,
        im.status
    FROM inference_model im
    JOIN inference_deployment idp ON im.inference_id = idp.id
    WHERE idp.is_deleted = FALSE AND im.is_deleted = FALSE;
    """

    async with db.acquire() as conn:
        services = await conn.fetch(query)
        return [
            {snake_to_camel(key): value for key, value in dict(service).items()}
            for service in services
        ]


from datetime import datetime


@router.patch("/model/{id}", response_model=dict)
async def delete_inference_model(
    id: int,
    request: Request,
):
    db = request.app.state.db_pool
    current_time = datetime.now()

    query = """
    UPDATE inference_model
    SET is_deleted = TRUE, deleted_at = $1, updated_at = $1
    WHERE id = $2 AND is_deleted = FALSE
    RETURNING id, model_name, visibility, inference_id, model_id, 
              max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, created_at, updated_at, deleted_at, is_deleted;
    """

    async with db.acquire() as conn:
        result = await conn.fetchrow(query, current_time, id)

        if not result:
            raise HTTPException(status_code=404, detail="Inference model not found")

        return {snake_to_camel(key): value for key, value in dict(result).items()}
