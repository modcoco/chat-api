import os
import asyncpg
import asyncio

from dotenv import load_dotenv


load_dotenv()

async def main():
    db_user = os.getenv("DB_USER", "default_user")
    db_password = os.getenv("DB_PASSWORD", "default_password")
    db_name = os.getenv("DB_NAME", "postgres")
    db_host = os.getenv("DB_HOST", "localhost")
    db_min_size = int(os.getenv("DB_MIN_SIZE", "1"))
    db_max_size = int(os.getenv("DB_MAX_SIZE", "10"))
    pool = await asyncpg.create_pool(
        user=db_user,
        password=db_password,
        database=db_name,
        host=db_host,
        min_size=db_min_size,
        max_size=db_max_size,
    )

    async with pool.acquire() as connection:
        records = await connection.fetch("SELECT * FROM models")

        for record in records:
            print(dict(record))

    await asyncio.sleep(10)  # 暂停 10 秒
    await pool.close()


asyncio.run(main())
