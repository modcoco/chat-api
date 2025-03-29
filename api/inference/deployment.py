from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request
import httpx

from app.inference_deployment import (
    create_inference_deployment,
    get_inference_deployments,
)
from models import InferenceDeployment, InferenceDeploymentCreate

router = APIRouter()


@router.post("/deployment")
async def create_deployment(
    request: Request,
    deployment: InferenceDeploymentCreate,
):
    db = request.app.state.db_pool
    async with db.acquire() as conn:
        existing_deployment = await conn.fetchrow(
            """
            SELECT * FROM inference_deployment 
            WHERE (inference_name = $1 OR deployment_url = $2) AND is_deleted = FALSE
            """,
            deployment.inference_name,
            deployment.deployment_url,
        )

        if existing_deployment:
            if existing_deployment["inference_name"] == deployment.inference_name:
                raise HTTPException(
                    status_code=400,
                    detail="Inference deployment with this name already exists.",
                )
            if existing_deployment["deployment_url"] == deployment.deployment_url:
                raise HTTPException(
                    status_code=400,
                    detail="Inference deployment with this URL already exists.",
                )

        new_deployment = await create_inference_deployment(conn, deployment)
        return new_deployment


@router.get("/deployment", response_model=List[InferenceDeployment])
async def get_deployments(
    request: Request,
    inference_name: Optional[str] = None,
):
    db = request.app.state.db_pool
    async with db.acquire() as conn:
        deployments = await get_inference_deployments(conn, inference_name)
        if not deployments:
            raise HTTPException(
                status_code=404, detail="Inference deployments not found."
            )
        return deployments


@router.patch("/deployment/{id}", status_code=204)
async def delete_deployment(
    request: Request,
    id: int,
):
    db = request.app.state.db_pool
    async with db.acquire() as conn:
        existing_deployment = await conn.fetchrow(
            "SELECT * FROM inference_deployment WHERE id = $1 AND is_deleted = FALSE",
            id,
        )

        if not existing_deployment:
            raise HTTPException(
                status_code=404,
                detail="Inference deployment not found or already deleted.",
            )

        await conn.execute(
            """
            UPDATE inference_deployment
            SET is_deleted = TRUE, deleted_at = $1, updated_at = $1, status = 'inactive'
            WHERE id = $2
            """,
            datetime.now(),
            id,
        )

    return {"detail": "Inference deployment marked as deleted successfully."}


@router.get("/deployment/{id}/models", response_model=List[dict])
async def get_model_ids(request: Request, id: int):
    db = request.app.state.db_pool
    async with db.acquire() as conn:
        deployment = await conn.fetchrow(
            """
            SELECT deployment_url, models_api_key 
            FROM inference_deployment 
            WHERE id = $1 AND is_deleted = FALSE
            """,
            id,
        )

        if not deployment:
            raise HTTPException(
                status_code=404, detail="Inference deployment not found."
            )

        deployment_url = deployment["deployment_url"]
        models_api_key = (
            deployment["models_api_key"] or "sk-default"
        )  # 默认使用 "sk-default" 如果为空
        url_with_models = f"{deployment_url}/v1/models"

        async with httpx.AsyncClient() as client:
            try:
                # 携带 Authorization 头
                headers = {"Authorization": f"Bearer {models_api_key}"}
                response = await client.get(url_with_models, headers=headers)
                response.raise_for_status()

                data = response.json()

                if isinstance(data, dict) and "data" in data:
                    # 返回模型列表，并关联 inference_id
                    filtered_data = [
                        {
                            "id": item.get("id"),
                            "inferenceId": id,
                            "created": item.get("created"),
                            "ownedBy": item.get("owned_by"),
                            "maxModelLen": item.get("max_model_len"),
                        }
                        for item in data["data"]
                    ]
                    return filtered_data
                else:
                    return []
            except httpx.RequestError as e:
                request.app.state.logger.error(f"Request error: {e}")
                return []
            except httpx.HTTPStatusError as e:
                request.app.state.logger.error(f"HTTP error: {e}")
                return []
