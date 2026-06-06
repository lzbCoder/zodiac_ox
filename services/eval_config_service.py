from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from models.rag_eval_config import RagEvalConfig


async def get_or_create_config(db: AsyncSession, kb_id: int) -> RagEvalConfig:
    stmt = select(RagEvalConfig).where(RagEvalConfig.kb_id == kb_id)
    config = (await db.execute(stmt)).scalar_one_or_none()
    if config:
        return config
    config = RagEvalConfig(kb_id=kb_id)
    db.add(config)
    await db.commit()
    await db.refresh(config)
    return config


async def update_config(db: AsyncSession, kb_id: int, default_top_k: int | None = None, default_retriever_mode: str | None = None) -> RagEvalConfig | None:
    config = await get_or_create_config(db, kb_id)
    if default_top_k is not None:
        config.default_top_k = default_top_k
    if default_retriever_mode is not None:
        config.default_retriever_mode = default_retriever_mode
    await db.commit()
    await db.refresh(config)
    return config
