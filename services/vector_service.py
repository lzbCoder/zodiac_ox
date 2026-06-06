from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pymilvus import Collection
from models.document import Document
from models.document_chunk import DocumentChunk


async def get_vector_stats(collection: Collection, db: AsyncSession) -> dict:
    collection.load()
    total = collection.num_entities

    doc_count_stmt = select(Document).where(Document.is_deleted == False)
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
    stmt = select(DocumentChunk.milvus_id).where(DocumentChunk.doc_id.in_(
        select(Document.id).where(Document.is_deleted == True)
    ))
    result = await db.execute(stmt)
    invalid_ids = [row[0] for row in result.all() if row[0]]
    if invalid_ids:
        collection.delete(f'id in {invalid_ids}')
