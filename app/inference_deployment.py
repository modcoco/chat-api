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
           TO_CHAR(created_at, 'YYYY-MM-DD"T"HH24:MI:SS') AS created_at, status  -- 格式化 created_at 字段为字符串
    FROM inference_deployment where is_deleted = FALSE
    """
    if inference_name:
        query += " WHERE inference_name = $1"
        rows = await conn.fetch(query, inference_name)
    else:
        rows = await conn.fetch(query)

    return [InferenceDeployment(**row) for row in rows]


async def create_inference_deployment(conn, deployment: InferenceDeploymentCreate):
    url_with_models = f"{deployment.deployment_url}/v1/models"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url_with_models)
            response.raise_for_status()  # 如果请求失败会抛出异常
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
    api_key: str, db: asyncpg.Pool
) -> Optional[Tuple[str, str]]:
    # 使用 JOIN 查询获取 deployment_url 和 models_api_key，明确指定每个表的 id 列
    query = """
    SELECT idp.deployment_url, idp.models_api_key
    FROM inference_deployment idp
    JOIN inference_model im ON idp.id = im.inference_id
    JOIN inference_model_api_key imak ON im.id = imak.inference_model_id
    WHERE imak.api_key = $1
    AND imak.is_deleted = FALSE
    """

    async with db.acquire() as conn:
        result = await conn.fetchrow(query, api_key)

    # 如果找到了匹配的结果，则返回 deployment_url 和 models_api_key，否则返回 None
    if result:
        deployment_url = result["deployment_url"]
        models_api_key = result["models_api_key"]
        if models_api_key is None:
            models_api_key = "please_set_models_api_key"
        return deployment_url, models_api_key
    else:
        return None
