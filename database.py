from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from config import DATABASE_URL
from loguru import logger

engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,    # 连接超过 1 小时后自动回收，防止服务端断开
    echo=False,
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# 评测任务专用连接池 — 与常规业务隔离，避免 RAGAS 拖慢 API 响应
eval_engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_timeout=120,
    echo=False,
)
async_eval_session = async_sessionmaker(eval_engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

            # Execute init_db.sql for indexes and supplemental DDL
            sql_path = Path(__file__).resolve().parent / "init_db.sql"
            if sql_path.exists():
                sql_text = sql_path.read_text(encoding="utf-8")
                for stmt in sql_text.split(";"):
                    stmt = stmt.strip()
                    # strip leading comment lines (a ; in a comment may split the block)
                    lines = stmt.split("\n")
                    while lines and (not lines[0].strip() or lines[0].strip().startswith("--")):
                        lines.pop(0)
                    stmt = "\n".join(lines).strip()
                    if stmt:
                        await conn.execute(text(stmt))
            else:
                logger.warning(f"init_db.sql not found at {sql_path}")
        logger.info("PostgreSQL tables initialized successfully.")
    except Exception as e:
        logger.warning(f"PostgreSQL init skipped (server unreachable): {e}")
