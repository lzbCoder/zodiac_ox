from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from models.knowledge_base import KnowledgeBase
from models.document import Document
from models.chat_history import ChatHistory
from models.rag_eval_dataset import RagEvalDataset
from models.rag_eval_label_task import RagEvalLabelTask


async def create_kb(db: AsyncSession, name: str, description: str | None = None) -> KnowledgeBase:
    kb = KnowledgeBase(name=name, description=description)
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return kb


async def list_kbs(db: AsyncSession) -> list[dict]:
    stmt = (
        select(
            KnowledgeBase,
            func.coalesce(func.count(Document.id), 0).label("doc_count"),
        )
        .outerjoin(Document, Document.kb_id == KnowledgeBase.id)
        .group_by(KnowledgeBase.id)
        .order_by(KnowledgeBase.updated_at.desc())
    )
    result = await db.execute(stmt)
    rows = result.all()
    return [
        {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description,
            "created_at": kb.created_at,
            "updated_at": kb.updated_at,
            "doc_count": doc_count,
            "vector_status": "normal",
        }
        for kb, doc_count in rows
    ]


async def get_kb(db: AsyncSession, kb_id: int) -> KnowledgeBase | None:
    stmt = select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_kb(db: AsyncSession, kb_id: int, name: str | None = None, description: str | None = None) -> KnowledgeBase | None:
    kb = await get_kb(db, kb_id)
    if not kb:
        return None
    if name is not None:
        kb.name = name
    if description is not None:
        kb.description = description
    await db.commit()
    await db.refresh(kb)
    return kb


async def delete_kb(db: AsyncSession, kb_id: int) -> bool:
    kb = await get_kb(db, kb_id)
    if not kb:
        return False
    # 手动清理无 FK 约束的关联表
    await db.execute(delete(ChatHistory).where(ChatHistory.kb_id == kb_id))
    await db.execute(delete(RagEvalDataset).where(RagEvalDataset.kb_id == kb_id))
    await db.execute(delete(RagEvalLabelTask).where(RagEvalLabelTask.kb_id == kb_id))
    # 删除知识库本身（使用 bulk delete 绕过 ORM cascade，DB 侧 ON DELETE CASCADE 自动清理子表）
    await db.execute(delete(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    await db.commit()
    return True
