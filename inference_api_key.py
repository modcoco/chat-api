from datetime import datetime
from typing import Optional
import asyncpg


async def get_api_key_id_by_key(api_key: str, db: asyncpg.Pool) -> Optional[int]:
    query = """
    SELECT id FROM inference_model_api_key WHERE api_key = $1 AND is_deleted = FALSE;
    """

    async with db.acquire() as conn:
        result = await conn.fetchrow(query, api_key)

    if result:
        return result["id"]
    return None


async def update_last_used_at(api_key: str, db: asyncpg.Pool) -> bool:
    current_time = datetime.now()

    query = """
    UPDATE inference_model_api_key
    SET last_used_at = $1
    WHERE api_key = $2 AND is_deleted = FALSE
    """

    async with db.acquire() as conn:
        result = await conn.execute(query, current_time, api_key)

    return result == "UPDATE 1"
