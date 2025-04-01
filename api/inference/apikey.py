from datetime import datetime, timedelta
from typing import List, Optional
import uuid
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi import status

from models import (
    ApiKeyTagResponse,
    DeleteApiKeyModelRelationResponse,
    ModelQuota,
    MultiModelApiKeyCreate,
    MultiModelApiKeyResponse,
    QuotaUpdateRequest,
    QuotaUpdateResponse,
)


router = APIRouter()


@router.post("/apikey", response_model=MultiModelApiKeyResponse)
async def create_multi_model_api_key(
    request: Request,
    api_key_data: MultiModelApiKeyCreate,
):
    db = request.app.state.db_pool
    unique_id = uuid.uuid4().hex
    api_key = f"sk-{unique_id}"

    # Calculate expiration time
    expires_at = None
    if api_key_data.active_days:
        expires_at = datetime.now() + timedelta(days=api_key_data.active_days)

    created_at_time = datetime.now()

    async with db.acquire() as conn:
        # Start transaction
        async with conn.transaction():
            # 1. Create main API Key record
            query_insert_key = """
            INSERT INTO inference_api_key (
                api_key_name, api_key, active_days, created_at, expires_at
            ) 
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, api_key_name, api_key, active_days, 
                      created_at, expires_at, is_deleted;
            """
            key_result = await conn.fetchrow(
                query_insert_key,
                api_key_data.api_key_name,
                api_key,
                api_key_data.active_days,
                created_at_time,
                expires_at,
            )

            # 2. Create quota records for each model
            key_id = key_result["id"]
            models = []

            for model_quota in api_key_data.model_quotas:
                # Check if model exists and is valid
                query_check_model = """
                SELECT im.id, im.model_name, idp.is_deleted as idp_is_deleted, idp.status
                FROM inference_model im
                JOIN inference_deployment idp ON im.inference_id = idp.id
                WHERE im.id = $1 AND im.is_deleted = FALSE
                """
                model_result = await conn.fetchrow(
                    query_check_model, model_quota.model_id
                )

                if not model_result:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Model with id {model_quota.model_id} not found or marked as deleted",
                    )

                if model_result["idp_is_deleted"]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"The associated inference deployment for model {model_quota.model_id} is marked as deleted",
                    )

                if model_result["status"] != "active":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"The associated inference deployment for model {model_quota.model_id} is not active",
                    )

                # Insert model quota record
                query_insert_quota = """
                INSERT INTO inference_api_key_model (
                    api_key_id, model_id, 
                    max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota,
                    created_at
                ) 
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id as relation_id, model_id, max_token_quota, max_prompt_tokens_quota, 
                          max_completion_tokens_quota, created_at, is_deleted;
                """
                quota_result = await conn.fetchrow(
                    query_insert_quota,
                    key_id,
                    model_quota.model_id,
                    model_quota.max_token_quota,
                    model_quota.max_prompt_tokens_quota,
                    model_quota.max_completion_tokens_quota,
                    created_at_time,
                )

                models.append(
                    {
                        "relationId": quota_result["relation_id"],  # Added
                        "model_id": quota_result["model_id"],
                        "model_name": model_result["model_name"],
                        "max_token_quota": quota_result["max_token_quota"],
                        "max_prompt_tokens_quota": quota_result[
                            "max_prompt_tokens_quota"
                        ],
                        "max_completion_tokens_quota": quota_result[
                            "max_completion_tokens_quota"
                        ],
                        "modelStatus": model_result["status"],  # Added
                        "created_at": quota_result["created_at"].isoformat(),
                        "is_deleted": quota_result["is_deleted"],
                    }
                )

    # Build response
    return {
        "id": key_result["id"],
        "api_key_name": key_result["api_key_name"],
        "api_key": key_result["api_key"],
        "active_days": key_result["active_days"],
        "created_at": key_result["created_at"].isoformat(),
        "expires_at": (
            key_result["expires_at"].isoformat() if key_result["expires_at"] else None
        ),
        "is_deleted": key_result["is_deleted"],
        "models": models,
        "tags": [],  # Added empty tags array if not implemented yet
    }


@router.post("/apikey/{api_key_id}/models", response_model=MultiModelApiKeyResponse)
async def add_models_to_api_key(
    request: Request,
    api_key_id: int,
    model_quotas: List[ModelQuota],
):
    db = request.app.state.db_pool
    created_at_time = datetime.now()

    async with db.acquire() as conn:
        # Start transaction
        async with conn.transaction():
            # 1. Check if API Key exists and is not deleted
            query_get_key = """
            SELECT id, api_key_name, api_key, active_days, 
                   created_at, expires_at, is_deleted
            FROM inference_api_key
            WHERE id = $1 AND is_deleted = FALSE
            """
            key_result = await conn.fetchrow(query_get_key, api_key_id)

            if not key_result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"API Key with id {api_key_id} not found or marked as deleted",
                )

            # 2. Check if the API Key is expired
            if key_result["expires_at"] and key_result["expires_at"] < datetime.now():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"API Key with id {api_key_id} has expired",
                )

            # 3. Get existing models for this API Key to check for duplicates
            query_existing_models = """
            SELECT model_id FROM inference_api_key_model
            WHERE api_key_id = $1 AND is_deleted = FALSE
            """
            existing_models = await conn.fetch(query_existing_models, api_key_id)
            existing_model_ids = {m["model_id"] for m in existing_models}

            models_to_add = []
            for model_quota in model_quotas:
                if model_quota.model_id in existing_model_ids:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Model with id {model_quota.model_id} is already associated with this API Key",
                    )
                models_to_add.append(model_quota)

            # 4. Add new model quotas
            added_models = []
            for model_quota in models_to_add:
                # Check if model exists and is valid
                query_check_model = """
                SELECT im.id, im.model_name, im.status,
                       idp.is_deleted as idp_is_deleted, idp.status as idp_status
                FROM inference_model im
                JOIN inference_deployment idp ON im.inference_id = idp.id
                WHERE im.id = $1 AND im.is_deleted = FALSE
                """
                model_result = await conn.fetchrow(
                    query_check_model, model_quota.model_id
                )

                if not model_result:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Model with id {model_quota.model_id} not found or marked as deleted",
                    )

                if model_result["idp_is_deleted"]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"The associated inference deployment for model {model_quota.model_id} is marked as deleted",
                    )

                if (
                    model_result["status"] != "active"
                    or model_result["idp_status"] != "active"
                ):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Model {model_quota.model_id} or its deployment is not active",
                    )

                # Insert model quota record
                query_insert_quota = """
                INSERT INTO inference_api_key_model (
                    api_key_id, model_id, 
                    max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota,
                    created_at
                ) 
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, model_id, max_token_quota, max_prompt_tokens_quota, 
                          max_completion_tokens_quota, created_at, is_deleted;
                """
                quota_result = await conn.fetchrow(
                    query_insert_quota,
                    api_key_id,
                    model_quota.model_id,
                    model_quota.max_token_quota,
                    model_quota.max_prompt_tokens_quota,
                    model_quota.max_completion_tokens_quota,
                    created_at_time,
                )

                added_models.append(
                    {
                        "relation_id": quota_result["id"],
                        "model_id": quota_result["model_id"],
                        "model_name": model_result["model_name"],
                        "max_token_quota": quota_result["max_token_quota"],
                        "max_prompt_tokens_quota": quota_result[
                            "max_prompt_tokens_quota"
                        ],
                        "max_completion_tokens_quota": quota_result[
                            "max_completion_tokens_quota"
                        ],
                        "created_at": quota_result["created_at"].isoformat(),
                        "is_deleted": quota_result["is_deleted"],
                        "used_prompt_tokens": 0,  # 默认值设为0
                        "used_completion_tokens": 0,
                        "used_total_tokens": 0,
                    }
                )

            # 5. Get all models for this API Key (existing + newly added)
            query_all_models = """
            SELECT 
                iakm.id as relation_id, 
                iakm.model_id, 
                im.model_name,
                iakm.max_token_quota, 
                iakm.max_prompt_tokens_quota, 
                iakm.max_completion_tokens_quota, 
                iakm.created_at,
                0 as used_prompt_tokens,  -- 默认值设为0
                0 as used_completion_tokens,
                0 as used_total_tokens
            FROM inference_api_key_model iakm
            JOIN inference_model im ON iakm.model_id = im.id
            WHERE iakm.api_key_id = $1 AND iakm.is_deleted = FALSE
            """
            all_models = await conn.fetch(query_all_models, api_key_id)

    # Build response
    return {
        "id": key_result["id"],
        "api_key_name": key_result["api_key_name"],
        "api_key": key_result["api_key"],
        "active_days": key_result["active_days"],
        "created_at": key_result["created_at"].isoformat(),
        "expires_at": (
            key_result["expires_at"].isoformat() if key_result["expires_at"] else None
        ),
        "is_deleted": key_result["is_deleted"],
        "models": [
            {
                "relation_id": m["relation_id"],
                "model_id": m["model_id"],
                "model_name": m["model_name"],
                "max_token_quota": m["max_token_quota"],
                "max_prompt_tokens_quota": m["max_prompt_tokens_quota"],
                "max_completion_tokens_quota": m["max_completion_tokens_quota"],
                "created_at": m["created_at"].isoformat(),
                "used_prompt_tokens": m["used_prompt_tokens"],
                "used_completion_tokens": m["used_completion_tokens"],
                "used_total_tokens": m["used_total_tokens"],
            }
            for m in all_models
        ],
    }


@router.delete(
    "/apikey/models/{relationId}",
    response_model=DeleteApiKeyModelRelationResponse,
)
async def delete_api_key_model_relation(
    request: Request,
    relationId: int,  # 使用 relationId 作为参数名
):
    db = request.app.state.db_pool
    deleted_at = datetime.now()

    async with db.acquire() as conn:
        async with conn.transaction():
            # 1. 检查关联记录是否存在且未删除
            query_check_relation = """
            SELECT id, api_key_id, model_id 
            FROM inference_api_key_model
            WHERE id = $1 AND is_deleted = FALSE
            """
            relation = await conn.fetchrow(query_check_relation, relationId)

            if not relation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Relation with id {relationId} not found or already deleted",
                )

            # 2. 检查关联的API Key是否有效
            query_check_key = """
            SELECT id FROM inference_api_key
            WHERE id = $1 AND is_deleted = FALSE
            """
            key_exists = await conn.fetchval(query_check_key, relation["api_key_id"])
            if not key_exists:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Associated API Key has been deleted",
                )

            # 3. 执行标记删除操作
            query_mark_deleted = """
            UPDATE inference_api_key_model
            SET is_deleted = TRUE, deleted_at = $1
            WHERE id = $2
            RETURNING id, api_key_id, model_id, deleted_at
            """
            deletion_result = await conn.fetchrow(
                query_mark_deleted, deleted_at, relationId
            )

    return {
        "success": True,
        "message": "API Key-Model relation successfully deleted",
        "deleted_at": deletion_result["deleted_at"].isoformat(),
        "api_key_id": deletion_result["api_key_id"],
        "model_id": deletion_result["model_id"],
        "relationId": deletion_result["id"],
    }


@router.patch("/api-key-model/{relation_id}/quotas", response_model=QuotaUpdateResponse)
async def update_api_key_model_quotas(
    request: Request, relation_id: int, quota_data: QuotaUpdateRequest
):
    """
    Update quota limits for an API key-model relation

    Parameters:
    - relation_id: The ID from inference_api_key_model table
    - quota_data: JSON body containing any of these optional fields:
        - max_token_quota
        - max_prompt_tokens_quota
        - max_completion_tokens_quota
        (pass -1 to set a quota to NULL)

    At least one quota field must be provided.
    """
    # Convert model to dict and remove unset fields
    update_data = quota_data.dict(exclude_unset=True)

    # Validate at least one field was provided
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one quota field must be provided for update",
        )

    # Build dynamic update query
    updates = []
    params = []

    # Add fields that were provided
    if "max_token_quota" in update_data:
        if update_data["max_token_quota"] == -1:
            updates.append("max_token_quota = NULL")
        else:
            updates.append("max_token_quota = $1")
            params.append(update_data["max_token_quota"])

    if "max_prompt_tokens_quota" in update_data:
        if update_data["max_prompt_tokens_quota"] == -1:
            updates.append("max_prompt_tokens_quota = NULL")
        else:
            position = len(params) + 1
            updates.append(f"max_prompt_tokens_quota = ${position}")
            params.append(update_data["max_prompt_tokens_quota"])

    if "max_completion_tokens_quota" in update_data:
        if update_data["max_completion_tokens_quota"] == -1:
            updates.append("max_completion_tokens_quota = NULL")
        else:
            position = len(params) + 1
            updates.append(f"max_completion_tokens_quota = ${position}")
            params.append(update_data["max_completion_tokens_quota"])

    # Add timestamp update
    updates.append("updated_at = NOW()")

    # Build final query
    query = f"""
        UPDATE inference_api_key_model
        SET {', '.join(updates)}
        WHERE id = ${len(params) + 1} AND is_deleted = FALSE
        RETURNING 
            id,
            api_key_id,
            model_id,
            max_token_quota,
            max_prompt_tokens_quota,
            max_completion_tokens_quota,
            updated_at;
    """
    params.append(relation_id)

    # Execute query
    db = request.app.state.db_pool
    async with db.acquire() as conn:
        result = await conn.fetchrow(query, *params)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key-model relation not found or already deleted",
            )

    return {
        "id": result["id"],
        "api_key_id": result["api_key_id"],
        "model_id": result["model_id"],
        "max_token_quota": result["max_token_quota"],
        "max_prompt_tokens_quota": result["max_prompt_tokens_quota"],
        "max_completion_tokens_quota": result["max_completion_tokens_quota"],
        "updated_at": result["updated_at"],
    }


@router.get("/apikey", response_model=List[MultiModelApiKeyResponse])
async def get_multi_model_api_keys(
    request: Request,
    tag: Optional[List[str]] = Query(None, description="Filter by tag names"),
):
    async with request.app.state.db_pool.acquire() as conn:
        # 基础CTE查询
        base_query = """
        WITH api_keys AS (
            SELECT id, api_key_name, api_key, active_days, 
                   created_at, last_used_at, expires_at
            FROM inference_api_key
            WHERE is_deleted = FALSE
        """

        # 如果有标签过滤条件
        if tag:
            base_query += """
            AND id IN (
                SELECT DISTINCT akta.api_key_id
                FROM api_key_tag_association akta
                JOIN api_key_tag akt ON akta.tag_id = akt.id
                WHERE akt.tag_name = ANY($1)
                AND akta.is_deleted = FALSE
                AND akt.is_deleted = FALSE
            )
            """

        base_query += """
            ORDER BY created_at DESC
        ),
        quotas AS (
            SELECT 
                iakm.id AS relation_id,
                iakm.api_key_id, 
                iakm.model_id, 
                im.model_name,
                im.status AS model_status,
                iakm.max_token_quota, 
                iakm.max_prompt_tokens_quota, 
                iakm.max_completion_tokens_quota, 
                iakm.created_at
            FROM inference_api_key_model iakm
            JOIN inference_model im ON iakm.model_id = im.id
            WHERE iakm.api_key_id IN (SELECT id FROM api_keys)
            AND iakm.is_deleted = FALSE
            AND im.is_deleted = FALSE
        ),
        key_tags AS (
            SELECT 
                akta.api_key_id,
                akt.id AS tag_id,
                akt.tag_name,
                akta.created_at AS tag_created_at
            FROM api_key_tag_association akta
            JOIN api_key_tag akt ON akta.tag_id = akt.id
            WHERE akta.api_key_id IN (SELECT id FROM api_keys)
            AND akta.is_deleted = FALSE
            AND akt.is_deleted = FALSE
        )
        SELECT 
            k.*,
            q.relation_id,
            q.model_id AS quota_model_id,
            q.model_name,
            q.model_status,
            q.max_token_quota,
            q.max_prompt_tokens_quota,
            q.max_completion_tokens_quota,
            q.created_at AS quota_created_at,
            kt.tag_id,
            kt.tag_name,
            kt.tag_created_at,
            COALESCE((
                SELECT SUM(prompt_tokens) 
                FROM inference_api_key_token_usage 
                WHERE api_key_id = k.id AND model_id = q.model_id
            ), 0) AS used_prompt_tokens,
            COALESCE((
                SELECT SUM(completion_tokens) 
                FROM inference_api_key_token_usage 
                WHERE api_key_id = k.id AND model_id = q.model_id
            ), 0) AS used_completion_tokens,
            COALESCE((
                SELECT SUM(prompt_tokens + completion_tokens) 
                FROM inference_api_key_token_usage 
                WHERE api_key_id = k.id AND model_id = q.model_id
            ), 0) AS used_total_tokens
        FROM api_keys k
        LEFT JOIN quotas q ON k.id = q.api_key_id
        LEFT JOIN key_tags kt ON k.id = kt.api_key_id
        ORDER BY k.created_at DESC, q.created_at, kt.tag_created_at;
        """

        # 执行查询（根据是否有标签参数决定参数传递）
        if tag:
            records = await conn.fetch(base_query, tag)
        else:
            records = await conn.fetch(base_query)

        # 按API Key分组数据（保持不变）
        response_map = {}
        for record in records:
            key_id = record["id"]

            if key_id not in response_map:
                response_map[key_id] = {
                    "id": key_id,
                    "api_key_name": record["api_key_name"],
                    "api_key": record["api_key"],
                    "active_days": record["active_days"],
                    "created_at": record["created_at"].isoformat(),
                    "last_used_at": (
                        record["last_used_at"].isoformat()
                        if record["last_used_at"]
                        else None
                    ),
                    "expires_at": (
                        record["expires_at"].isoformat()
                        if record["expires_at"]
                        else None
                    ),
                    "models": [],
                    "tags": [],
                }

            # 添加模型信息
            if record["quota_model_id"] and not any(
                m["model_id"] == record["quota_model_id"]
                for m in response_map[key_id]["models"]
            ):
                response_map[key_id]["models"].append(
                    {
                        "relation_id": record["relation_id"],
                        "model_id": record["quota_model_id"],
                        "model_name": record["model_name"],
                        "model_status": record["model_status"],
                        "max_token_quota": record["max_token_quota"],
                        "max_prompt_tokens_quota": record["max_prompt_tokens_quota"],
                        "max_completion_tokens_quota": record[
                            "max_completion_tokens_quota"
                        ],
                        "used_prompt_tokens": record["used_prompt_tokens"],
                        "used_completion_tokens": record["used_completion_tokens"],
                        "used_total_tokens": record["used_total_tokens"],
                        "created_at": record["quota_created_at"].isoformat(),
                    }
                )

            # 添加标签信息
            if record["tag_id"] and not any(
                t["tag_id"] == record["tag_id"] for t in response_map[key_id]["tags"]
            ):
                response_map[key_id]["tags"].append(
                    {
                        "tag_id": record["tag_id"],
                        "tag_name": record["tag_name"],
                        "created_at": record["tag_created_at"].isoformat(),
                    }
                )

    return list(response_map.values())


@router.patch("/apikey/{api_key_id}/delete")
async def delete_multi_model_api_key(
    request: Request,
    api_key_id: int,
):
    db = request.app.state.db_pool
    async with db.acquire() as conn:
        # Start a transaction
        async with conn.transaction():
            # Check if API key exists and is not already deleted
            result = await conn.fetchrow(
                "SELECT id FROM inference_api_key WHERE id = $1 AND is_deleted = FALSE;",
                api_key_id,
            )
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="API key not found or already deleted",
                )

            # Mark all associated models as deleted first
            await conn.execute(
                """
                UPDATE inference_api_key_model
                SET is_deleted = TRUE, deleted_at = NOW()
                WHERE api_key_id = $1 AND is_deleted = FALSE;
                """,
                api_key_id,
            )

            # Then mark the API key as deleted
            updated_result = await conn.fetchrow(
                """
                UPDATE inference_api_key
                SET is_deleted = TRUE, deleted_at = NOW()
                WHERE id = $1
                RETURNING id, is_deleted;
                """,
                api_key_id,
            )

    return {"id": updated_result["id"], "isDeleted": updated_result["is_deleted"]}


@router.post("/apikey/{api_key_id}/tags", response_model=ApiKeyTagResponse)
async def add_tags_to_api_key(
    request: Request,
    api_key_id: int,
    tag_names: List[str],
):
    """
    为API Key添加标签
    """
    db = request.app.state.db_pool
    created_at = datetime.now()

    async with db.acquire() as conn:
        async with conn.transaction():
            # 1. 检查API Key是否存在且未删除
            query_check_key = """
            SELECT id FROM inference_api_key
            WHERE id = $1 AND is_deleted = FALSE
            """
            key_exists = await conn.fetchval(query_check_key, api_key_id)
            if not key_exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"API Key with id {api_key_id} not found or marked as deleted",
                )

            # 2. 处理每个标签
            added_tags = []
            for tag_name in tag_names:
                # 2.1 检查标签是否存在，不存在则创建
                query_get_or_create_tag = """
                WITH new_tag AS (
                    INSERT INTO api_key_tag (tag_name, created_at)
                    VALUES ($1, $2)
                    ON CONFLICT (tag_name) DO NOTHING
                    RETURNING id, tag_name, created_at
                )
                SELECT id, tag_name, created_at FROM new_tag
                UNION
                SELECT id, tag_name, created_at FROM api_key_tag
                WHERE tag_name = $1 AND is_deleted = FALSE
                """
                tag_result = await conn.fetchrow(
                    query_get_or_create_tag,
                    tag_name,
                    created_at,
                )

                # 2.2 检查是否已关联
                query_check_association = """
                SELECT 1 FROM api_key_tag_association
                WHERE api_key_id = $1 AND tag_id = $2 AND is_deleted = FALSE
                """
                association_exists = await conn.fetchval(
                    query_check_association,
                    api_key_id,
                    tag_result["id"],
                )

                if association_exists:
                    continue  # 已存在关联则跳过

                # 2.3 创建关联
                query_create_association = """
                INSERT INTO api_key_tag_association (
                    api_key_id, tag_id, created_at
                )
                VALUES ($1, $2, $3)
                ON CONFLICT (api_key_id, tag_id) 
                DO UPDATE SET is_deleted = FALSE, deleted_at = NULL
                """
                await conn.execute(
                    query_create_association,
                    api_key_id,
                    tag_result["id"],
                    created_at,
                )

                added_tags.append(
                    {
                        "tag_id": tag_result["id"],
                        "tag_name": tag_result["tag_name"],
                        "created_at": tag_result["created_at"].isoformat(),
                    }
                )

    return {
        "api_key_id": api_key_id,
        "added_tags": added_tags,
        "message": "Tags added successfully" if added_tags else "No new tags added",
    }


@router.delete("/apikey/{api_key_id}/tags", response_model=ApiKeyTagResponse)
async def remove_tags_from_api_key(
    request: Request,
    api_key_id: int,
    tag_names: List[str],
):
    """
    从API Key移除标签
    """
    db = request.app.state.db_pool
    deleted_at = datetime.now()

    async with db.acquire() as conn:
        async with conn.transaction():
            # 1. 检查API Key是否存在且未删除
            query_check_key = """
            SELECT id FROM inference_api_key
            WHERE id = $1 AND is_deleted = FALSE
            """
            key_exists = await conn.fetchval(query_check_key, api_key_id)
            if not key_exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"API Key with id {api_key_id} not found or marked as deleted",
                )

            # 2. 处理每个标签
            removed_tags = []
            for tag_name in tag_names:
                # 2.1 获取标签ID
                query_get_tag = """
                SELECT id, tag_name FROM api_key_tag
                WHERE tag_name = $1 AND is_deleted = FALSE
                """
                tag_result = await conn.fetchrow(query_get_tag, tag_name)
                if not tag_result:
                    continue  # 标签不存在则跳过

                # 2.2 标记删除关联
                query_delete_association = """
                UPDATE api_key_tag_association
                SET is_deleted = TRUE, deleted_at = $1
                WHERE api_key_id = $2 AND tag_id = $3 AND is_deleted = FALSE
                """
                await conn.execute(
                    query_delete_association,
                    deleted_at,
                    api_key_id,
                    tag_result["id"],
                )

                removed_tags.append(
                    {
                        "tag_id": tag_result["id"],
                        "tag_name": tag_result["tag_name"],
                        "deleted_at": deleted_at.isoformat(),
                    }
                )

    return {
        "api_key_id": api_key_id,
        "removed_tags": removed_tags,
        "message": "Tags removed successfully" if removed_tags else "No tags removed",
    }
