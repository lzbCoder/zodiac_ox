from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from models.knowledge_base import KnowledgeBase
from models.document import Document


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
        .outerjoin(Document, (Document.kb_id == KnowledgeBase.id) & (Document.is_deleted == False))
        .where(KnowledgeBase.is_deleted == False)
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
            "is_deleted": kb.is_deleted,
            "doc_count": doc_count,
            "vector_status": "normal",
        }
        for kb, doc_count in rows
    ]


async def get_kb(db: AsyncSession, kb_id: int) -> KnowledgeBase | None:
    stmt = select(KnowledgeBase).where(KnowledgeBase.id == kb_id, KnowledgeBase.is_deleted == False)
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
    kb.is_deleted = True
    await db.commit()
    return True
