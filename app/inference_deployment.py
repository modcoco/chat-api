from datetime import datetime
from typing import List, Optional, Tuple

import asyncpg
import httpx

from models import InferenceDeployment, InferenceDeploymentCreate


async def get_inference_deployments(
    conn, inference_name: Optional[str] = None
) -> List[InferenceDeployment]:
    query = """
    SELECT id, inference_name, type, deployment_url, models_api_key, 
           TO_CHAR(created_at, 'YYYY-MM-DD"T"HH24:MI:SS') AS created_at, status
    FROM inference_deployment where is_deleted = FALSE
    """
    if inference_name:
        query += " AND inference_name = $1"
        rows = await conn.fetch(query, inference_name)
    else:
        rows = await conn.fetch(query)

    return [InferenceDeployment(**row) for row in rows]


async def update_deployment_status(conn, deployment_id: int, status: str):
    query = """
    UPDATE inference_deployment
    SET status = $1, updated_at = NOW()
    WHERE id = $2
    """
    await conn.execute(query, status, deployment_id)


async def create_inference_deployment(conn, deployment: InferenceDeploymentCreate):
    url_with_models = f"{deployment.deployment_url}/v1/models"

    api_key = (
        deployment.models_api_key
        if deployment.models_api_key is not None
        else "sk-default"
    )
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url_with_models, headers=headers)
            response.raise_for_status()
            data = response.json()

            if (
                isinstance(data, dict)
                and "data" in data
                and isinstance(data["data"], list)
                and len(data["data"]) >= 0
            ):
                status = "active"
            else:
                status = "inactive"
        except httpx.RequestError as e:
            status = "inactive"
        except httpx.HTTPStatusError as e:
            status = "inactive"
        except Exception as e:
            status = "inactive"

    query = """
    INSERT INTO inference_deployment (inference_name, type, deployment_url, models_api_key, created_at, status)
    VALUES ($1, $2, $3, $4, $5, $6)
    RETURNING id, inference_name, type, deployment_url, models_api_key, created_at, status;
    """
    result = await conn.fetchrow(
        query,
        deployment.inference_name,
        deployment.type,
        deployment.deployment_url,
        deployment.models_api_key,
        datetime.now(),
        status,
    )
    return dict(result)


async def get_deployment_info_by_api_key(
    api_key: str, model_id: int, db: asyncpg.Pool  # 新增的 model_id 参数
) -> Optional[Tuple[str, str]]:
    """
    根据API Key和model_id获取特定模型的部署信息

    参数:
        api_key: 用户提供的API Key
        model_id: 请求的模型ID
        db: 数据库连接池

    返回:
        元组 (deployment_url, models_api_key) 或 None
    """
    query = """
    SELECT idp.deployment_url, idp.models_api_key
    FROM inference_deployment idp
    JOIN inference_model im ON idp.id = im.inference_id
    JOIN inference_api_key_model iakm ON im.id = iakm.model_id
    JOIN inference_api_key iak ON iakm.api_key_id = iak.id
    WHERE iak.api_key = $1
    AND im.id = $2
    AND iak.is_deleted = FALSE
    AND im.is_deleted = FALSE
    """

    async with db.acquire() as conn:
        result = await conn.fetchrow(query, api_key, model_id)

    if result:
        deployment_url = result["deployment_url"]
        models_api_key = result["models_api_key"] or "please_set_models_api_key"
        return deployment_url, models_api_key
    return None
