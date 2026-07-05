from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pymilvus import Collection
from models.document import Document


async def get_vector_stats(collection: Collection, db: AsyncSession) -> dict:
    collection.load()
    total = collection.num_entities

    doc_count_stmt = select(Document)
    result = await db.execute(doc_count_stmt)
    docs = result.scalars().all()
    doc_count = len(docs)

    return {
        "total_vectors": total,
        "valid_vectors": total,
        "doc_count": doc_count,
        "collection_name": collection.name,
    }


async def clean_invalid_vectors(collection: Collection, db: AsyncSession):
    # 物理删除后不需要清理已删文档的向量 — 文档行已不存在
    pass
