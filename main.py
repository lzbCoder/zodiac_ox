import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.formparsers import MultiPartParser
from log_config import setup_logger
from loguru import logger
from database import init_db, async_session
from milvus_client import connect_milvus, init_milvus_collection, disconnect_milvus
from redis_client import get_redis, close_redis
from otel_tracer import init_otel_tracer

# Starlette 底层限制：单文件最大 10M
MultiPartParser.max_part_size = 10 * 1024 * 1024

setup_logger()

# Initialize OTel BEFORE importing routers — ensures TracerProvider is set
# before any module-level `tracer = trace.get_tracer(__name__)` runs.
init_otel_tracer()

from routers import knowledge_base, document, chat, history, vector, system, rag_eval, monitor
from services.memory_service import init_memory_manager
from database import engine
from config import PG_SCHEMA


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------- infrastructure (no OTel dependency) ----------
    try:
        connect_milvus()
        init_milvus_collection()
    except Exception as e:
        logger.warning(f"Milvus init skipped: {e}")
    await init_db()

    # Initialize enterprise memory manager (dual-layer: session + user global)
    init_memory_manager(async_engine=engine, db_schema=PG_SCHEMA)
    logger.info("EnterpriseMemoryManager initialized")

    # Apply OTel config from DB BEFORE instrumenting routes —
    # ensures disabled state is respected on first startup.
    from otel_tracer import apply_otel_config_from_db
    await apply_otel_config_from_db()

    # ---------- OTel FastAPI instrumentation (after DB config) ----------
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI OTel instrumentation active")
    except Exception as e:
        logger.warning(f"FastAPI instrumentation skipped: {e}")
    try:
        await get_redis()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis init skipped: {e}")

    # Start background flush task
    flush_task = asyncio.create_task(_periodic_flush())

    yield

    flush_task.cancel()
    try:
        await flush_task
    except asyncio.CancelledError:
        pass
    disconnect_milvus()
    await close_redis()


async def _periodic_flush():
    """Flush trace data from Redis to PostgreSQL every 30 seconds."""
    while True:
        await asyncio.sleep(30)
        try:
            async with async_session() as db:
                from services import monitor_service
                await monitor_service.flush_traces(db)
        except Exception as e:
            logger.error(f"Periodic flush failed: {e}")


MAX_REQUEST_BODY = 100 * 1024 * 1024  # 100M — FastAPI 全局最大请求体
MAX_UPLOAD_SIZE = 10 * 1024 * 1024   # 10M — 单文件业务限制


app = FastAPI(
    title="越群山知识库RAG系统",
    version="0.1.0",
    lifespan=lifespan,
)


# Body-size guard — reject oversized requests before parsing (100M gateway limit)
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BODY:
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=413,
            content={"detail": f"请求体超过最大限制 {MAX_REQUEST_BODY // 1024 // 1024}M"},
        )
    return await call_next(request)


# Access log middleware — routes every request through Loguru
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({elapsed_ms:.0f}ms)"
    )
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(knowledge_base.router)
app.include_router(document.router)
app.include_router(chat.router)
app.include_router(history.router)
app.include_router(vector.router)
app.include_router(system.router)
app.include_router(rag_eval.router)
app.include_router(monitor.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
