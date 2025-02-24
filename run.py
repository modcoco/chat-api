import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import os
from typing import List, Optional
import uuid
import asyncpg
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from grpc import Status
import httpx

from app.inference_deployment import (
    create_inference_deployment,
    get_inference_deployments,
)
from app.inference_model import create_inference_model
from app.queue import process_token_usage_queue
from models import (
    InferenceDeployment,
    InferenceDeploymentCreate,
    InferenceModelApiKeyCreate,
    InferenceModelApiKeyResponse,
    InferenceModelCreate,
)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_user = os.getenv("DB_USER", "default_user")
    db_password = os.getenv("DB_PASSWORD", "default_password")
    db_name = os.getenv("DB_NAME", "postgres")
    db_host = os.getenv("DB_HOST", "localhost")
    db_min_size = int(os.getenv("DB_MIN_SIZE", "1"))
    db_max_size = int(os.getenv("DB_MAX_SIZE", "10"))

    print("Initializing database connection pool...")
    pool = await asyncpg.create_pool(
        user=db_user,
        password=db_password,
        database=db_name,
        host=db_host,
        min_size=db_min_size,
        max_size=db_max_size,
    )
    app.state.db_pool = pool

    app.state.token_usage_queue = asyncio.Queue()
    asyncio.create_task(process_token_usage_queue(app))

    yield
    await pool.close()
    print("Closing database connection pool...")


app = FastAPI(lifespan=lifespan)

from api.openai import router as chat_router

app.include_router(chat_router)


@app.post("/inference-deployment")
async def create_deployment(
    deployment: InferenceDeploymentCreate,
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
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


@app.get("/inference-deployment", response_model=List[InferenceDeployment])
async def get_deployments(
    inference_name: Optional[str] = None,
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
    async with db.acquire() as conn:
        deployments = await get_inference_deployments(conn, inference_name)
        if not deployments:
            raise HTTPException(
                status_code=404, detail="Inference deployments not found."
            )
        return deployments


@app.delete("/inference-deployment/{id}", status_code=204)
async def delete_deployment(
    id: int,
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
    async with db.acquire() as conn:
        existing_deployment = await conn.fetchrow(
            "SELECT * FROM inference_deployment WHERE id = $1", id
        )
        if not existing_deployment:
            raise HTTPException(
                status_code=404, detail="Inference deployment not found."
            )

        await conn.execute("DELETE FROM inference_deployment WHERE id = $1", id)

        return {"detail": "Inference deployment deleted successfully."}


@app.get("/inference-deployment/{id}/models", response_model=List[dict])
async def get_model_ids(id: int, db: asyncpg.Pool = Depends(lambda: app.state.db_pool)):
    async with db.acquire() as conn:
        deployment = await conn.fetchrow(
            "SELECT deployment_url FROM inference_deployment WHERE id = $1", id
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


# API
@app.post("/inference-model", response_model=InferenceModelCreate)
async def add_inference_model(
    model: InferenceModelCreate, db: asyncpg.Pool = Depends(lambda: app.state.db_pool)
):
    async with db.acquire() as conn:
        inserted_model = await create_inference_model(conn, model)
        return JSONResponse(content=inserted_model, status_code=201)


@app.get("/inference-model", response_model=List[dict])
async def get_all_inference_services(
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
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
        im.created,
        im.updated
    FROM inference_model im
    JOIN inference_deployment idp ON im.inference_id = idp.id;
    """

    async with db.acquire() as conn:
        services = await conn.fetch(query)
        return [dict(service) for service in services]


@app.delete("/inference-model/{id}", response_model=dict)
async def delete_inference_model(
    id: int, db: asyncpg.Pool = Depends(lambda: app.state.db_pool)
):
    query = "DELETE FROM inference_model WHERE id = $1 RETURNING id, model_name, visibility, inference_id, model_id, max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, created, updated;"

    async with db.acquire() as conn:
        result = await conn.fetchrow(query, id)

        if not result:
            raise HTTPException(status_code=404, detail="Inference model not found")

        return dict(result)


@app.post("/api-key", response_model=dict)
async def create_inference_model_api_key(
    api_key_data: InferenceModelApiKeyCreate,
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
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


@app.get("/api-keys", response_model=List[InferenceModelApiKeyResponse])
async def get_inference_model_api_keys(
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
    query = """
    SELECT id, api_key_name, inference_model_id, api_key, 
           max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota,
           active_days, created, last_used_at, expires_at, is_deleted
    FROM inference_model_api_key
    WHERE is_deleted = FALSE;
    """

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


@app.patch("/api-key/{api_key_id}/delete")
async def delete_inference_model_api_key(
    api_key_id: int,
    db: asyncpg.Pool = Depends(lambda: app.state.db_pool),
):
    query = """
    SELECT id FROM inference_model_api_key WHERE id = $1 AND is_deleted = FALSE;
    """
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
