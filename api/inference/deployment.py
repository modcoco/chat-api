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
            "SELECT * FROM inference_deployment WHERE inference_name = $1",
            deployment.inference_name,
        )
        if existing_deployment:
            raise HTTPException(
                status_code=400, detail="Inference deployment already exists."
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
            SET is_deleted = TRUE, deleted_at = $1, updated_at = $1
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
            "SELECT deployment_url FROM inference_deployment WHERE id = $1 AND is_deleted = FALSE",
            id,
        )

        if not deployment:
            raise HTTPException(
                status_code=404, detail="Inference deployment not found."
            )

        deployment_url = deployment["deployment_url"]
        url_with_models = f"{deployment_url}/v1/models"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url_with_models)
                response.raise_for_status()

                data = response.json()

                if "data" in data:
                    filtered_data = [
                        {
                            "id": item.get("id"),
                            "inference_id": id,
                            "created": item.get("created"),
                            "owned_by": item.get("owned_by"),
                            "max_model_len": item.get("max_model_len"),
                        }
                        for item in data["data"]
                    ]
                    return filtered_data
                else:
                    return []
            except httpx.RequestError as e:
                return []
            except httpx.HTTPStatusError as e:
                return []
