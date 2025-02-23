from datetime import datetime
from typing import List, Optional

import httpx

from models import InferenceDeployment, InferenceDeploymentCreate


async def get_inference_deployments(
    conn, inference_name: Optional[str] = None
) -> List[InferenceDeployment]:
    query = """
    SELECT id, inference_name, type, deployment_url, models_api_key, 
           TO_CHAR(created, 'YYYY-MM-DD"T"HH24:MI:SS') AS created, status  -- 格式化 created 字段为字符串
    FROM inference_deployment
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
    INSERT INTO inference_deployment (inference_name, type, deployment_url, models_api_key, created, status)
    VALUES ($1, $2, $3, $4, $5, $6)
    RETURNING id, inference_name, type, deployment_url, models_api_key, created, status;
    """
    result = await conn.fetchrow(
        query,
        deployment.inference_name,
        deployment.type,
        deployment.deployment_url,
        deployment.models_api_key,
        datetime.now(),  # 自动填充 created 字段
        status,  # 根据检查的结果设置 status
    )
    return dict(result)
