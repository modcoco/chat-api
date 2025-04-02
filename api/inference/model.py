import asyncio
from datetime import datetime
from functools import lru_cache
from typing import Dict, List
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx

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


@lru_cache(maxsize=100)
def should_check_model(model_id: str) -> bool:
    """基于模型ID决定是否需要检查状态（60秒缓存）"""
    return True


async def check_model_health(deployment_url: str, api_key: str, model_id: str) -> bool:
    """检查模型健康状态"""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                f"{deployment_url.rstrip('/')}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            models = resp.json()
            return any(
                m.get("id", "").endswith(model_id) for m in models.get("data", [])
            )
    except Exception as e:
        print(f"Model health check failed: {str(e)}")
        return False


@router.get("/model", response_model=List[Dict])
async def get_all_inference_services(request: Request):
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
        im.status,
        idp.deployment_url,
        idp.models_api_key
    FROM inference_model im
    JOIN inference_deployment idp ON im.inference_id = idp.id
    WHERE idp.is_deleted = FALSE AND im.is_deleted = FALSE;
    """

    async with db.acquire() as conn:
        services = await conn.fetch(query)
        results = []

        for service in services:
            service_dict = dict(service)
            camel_case_dict = {snake_to_camel(k): v for k, v in service_dict.items()}
            results.append(camel_case_dict)

            # 异步触发状态检查（仅对active状态且需要检查的模型）
            if service_dict["status"] == "active" and should_check_model(
                service_dict["model_id"]
            ):
                asyncio.create_task(update_model_status(db, service_dict))

        return results


async def update_model_status(db_pool, service_data: Dict):
    """异步更新模型状态"""
    try:
        async with db_pool.acquire() as conn:
            is_healthy = await check_model_health(
                service_data["deployment_url"],
                service_data["models_api_key"],
                service_data["model_id"],
            )

            # 当状态实际变化时才更新数据库
            new_status = "active" if is_healthy else "inactive"
            if new_status != service_data["status"]:
                await conn.execute(
                    "UPDATE inference_model SET status = $1 WHERE id = $2",
                    new_status,
                    service_data["id"],
                )
                print(
                    f"Updated model {service_data['model_name']} status to {new_status}"
                )

    except Exception as e:
        print(f"Failed to update model status: {str(e)}")


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
