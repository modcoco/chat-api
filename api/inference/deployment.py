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
        # 1. 获取部署信息（带状态）
        deployment = await conn.fetchrow(
            """
            SELECT id, deployment_url, models_api_key, status
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
        models_api_key = deployment["models_api_key"] or "sk-default"
        current_status = deployment["status"]
        url_with_models = f"{deployment_url.rstrip('/')}/v1/models"

        # 2. 尝试请求模型列表
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = {"Authorization": f"Bearer {models_api_key}"}
                response = await client.get(url_with_models, headers=headers)
                response.raise_for_status()
                data = response.json()

                # 3. 请求成功 - 更新状态为active（如果之前不是）
                if current_status != "active":
                    await conn.execute(
                        "UPDATE inference_deployment SET status = 'active' WHERE id = $1",
                        id,
                    )

                # 4. 返回模型列表
                if isinstance(data, dict) and "data" in data:
                    return [
                        {
                            "id": item.get("id"),
                            "inferenceId": id,
                            "created": item.get("created"),
                            "ownedBy": item.get("owned_by"),
                            "maxModelLen": item.get("max_model_len"),
                        }
                        for item in data["data"]
                    ]
                return []

        except Exception as e:
            # 5. 请求失败处理
            print(f"Failed to fetch models: {str(e)}")

            # 6. 更新状态为inactive（如果之前不是）
            if current_status != "inactive":
                await conn.execute(
                    "UPDATE inference_deployment SET status = 'inactive' WHERE id = $1",
                    id,
                )

            # 7. 返回错误信息（可选择返回空列表或错误详情）
            raise HTTPException(
                status_code=424,  # Failed Dependency
                detail=f"Deployment is currently unavailable. Status set to inactive. Original error: {str(e)}",
            )
