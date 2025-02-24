import asyncio
from contextlib import asynccontextmanager
import os
import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI
from app.queue import process_token_usage_queue

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
from api.inference.deployment import router as deployment_router
from api.inference.model import router as model_router

app.include_router(chat_router)
app.include_router(deployment_router)
app.include_router(model_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
