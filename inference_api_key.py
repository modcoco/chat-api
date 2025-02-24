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
