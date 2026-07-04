import os
import aiofiles
import random
from datetime import datetime
from pathlib import Path
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from models.document import Document
from models.document_chunk import DocumentChunk
from models.chunk_config import ChunkConfig
from models.knowledge_base import KnowledgeBase
from config import DATA_DIR


async def save_upload_file(file, kb_id: int) -> tuple[str, str, int]:
    kb_dir = DATA_DIR / str(kb_id)
    kb_dir.mkdir(parents=True, exist_ok=True)
    file_ext = Path(file.filename).suffix.lower()
    original_name = Path(file.filename).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = f"{random.randint(0, 99):02d}"
    stored_name = f"{original_name}_{timestamp}_{random_suffix}{file_ext}"
    file_path = kb_dir / stored_name
    content = await file.read()
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)
    return str(file_path), file_ext.lstrip("."), len(content)


def delete_local_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)


async def create_document(db: AsyncSession, kb_id: int, filename: str, file_type: str, file_path: str, file_size: int) -> Document:
    doc = Document(kb_id=kb_id, filename=filename, file_type=file_type, file_path=file_path, file_size=file_size, upload_status="pending")
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def get_document(db: AsyncSession, doc_id: int) -> Document | None:
    stmt = select(Document).where(Document.id == doc_id, Document.is_deleted == False)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def list_documents(db: AsyncSession, kb_id: int | None = None, file_type: str | None = None, filename: str | None = None, page: int = 1, page_size: int = 20) -> tuple[list[Document], int]:
    base = select(Document).where(Document.is_deleted == False)
    count_stmt = select(func.count(Document.id)).where(Document.is_deleted == False)

    if kb_id:
        base = base.where(Document.kb_id == kb_id)
        count_stmt = count_stmt.where(Document.kb_id == kb_id)
    if file_type:
        base = base.where(Document.file_type == file_type)
        count_stmt = count_stmt.where(Document.file_type == file_type)
    if filename:
        pattern = f"%{filename}%"
        base = base.where(Document.filename.ilike(pattern))
        count_stmt = count_stmt.where(Document.filename.ilike(pattern))

    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = (
        select(Document, KnowledgeBase.name)
        .join(KnowledgeBase, Document.kb_id == KnowledgeBase.id)
        .where(Document.is_deleted == False)
    )
    if kb_id:
        stmt = stmt.where(Document.kb_id == kb_id)
    if file_type:
        stmt = stmt.where(Document.file_type == file_type)
    if filename:
        stmt = stmt.where(Document.filename.ilike(pattern))
    stmt = stmt.order_by(Document.created_at.desc()).offset((page - 1) * page_size).limit(page_size)

    rows = (await db.execute(stmt)).all()
    docs = []
    for doc, kb_name in rows:
        doc.kb_name = kb_name
        docs.append(doc)
    return docs, total


async def update_document_status(db: AsyncSession, doc_id: int, upload_status: str | None = None, vector_status: str | None = None, chunk_count: int | None = None):
    doc = await get_document(db, doc_id)
    if not doc:
        return
    if upload_status is not None:
        doc.upload_status = upload_status
    if vector_status is not None:
        doc.vector_status = vector_status
    if chunk_count is not None:
        doc.chunk_count = chunk_count
    await db.commit()


async def delete_document(db: AsyncSession, doc_id: int) -> bool:
    doc = await get_document(db, doc_id)
    if not doc:
        return False
    doc.is_deleted = True
    await db.commit()
    return True


async def get_or_create_chunk_config(db: AsyncSession, kb_id: int) -> ChunkConfig:
    stmt = select(ChunkConfig).where(ChunkConfig.kb_id == kb_id)
    result = await db.execute(stmt)
    config = result.scalar_one_or_none()
    if not config:
        config = ChunkConfig(kb_id=kb_id)
        db.add(config)
        await db.commit()
        await db.refresh(config)
    return config


async def update_chunk_config(db: AsyncSession, kb_id: int, chunk_size: int, chunk_overlap: int, split_separator: str) -> ChunkConfig:
    config = await get_or_create_chunk_config(db, kb_id)
    config.chunk_size = chunk_size
    config.chunk_overlap = chunk_overlap
    config.split_separator = split_separator
    await db.commit()
    await db.refresh(config)
    return config


async def save_chunks(db: AsyncSession, kb_id: int, doc_id: int, chunks: list[dict]) -> list[DocumentChunk]:
    chunk_objs = []
    for c in chunks:
        chunk = DocumentChunk(
            kb_id=kb_id,
            doc_id=doc_id,
            content=c["content"],
            chunk_index=c["chunk_index"],
            page_num=c.get("page_num", 0),
            start_pos=c.get("start_pos", 0),
            end_pos=c.get("end_pos", 0),
            milvus_id=c.get("milvus_id"),
        )
        db.add(chunk)
        chunk_objs.append(chunk)
    await db.commit()
    return chunk_objs


async def get_distinct_file_types(db: AsyncSession) -> list[str]:
    """查询文档表中所有已出现的文件类型（去重）。"""
    from sqlalchemy import select, func
    stmt = select(Document.file_type).distinct().where(Document.is_deleted == False).order_by(Document.file_type)
    rows = (await db.execute(stmt)).scalars().all()
    return list(rows)


async def get_chunks_by_doc(db: AsyncSession, doc_id: int) -> list[DocumentChunk]:
    stmt = select(DocumentChunk).where(DocumentChunk.doc_id == doc_id).order_by(DocumentChunk.chunk_index)
    result = await db.execute(stmt)
    return result.scalars().all()


async def update_chunks_milvus_ids(db: AsyncSession, chunk_ids: list[int], milvus_ids: list[str]):
    from sqlalchemy import update, case
    if not chunk_ids:
        return
    # 单条 CASE WHEN bulk UPDATE 替代 N 次独立 UPDATE
    whens = [(DocumentChunk.id == cid, mid) for cid, mid in zip(chunk_ids, milvus_ids)]
    stmt = (
        update(DocumentChunk)
        .where(DocumentChunk.id.in_(chunk_ids))
        .values(milvus_id=case(*whens))
    )
    await db.execute(stmt)
    await db.commit()


async def delete_chunks_by_doc(db: AsyncSession, doc_id: int):
    from sqlalchemy import delete
    stmt = delete(DocumentChunk).where(DocumentChunk.doc_id == doc_id)
    await db.execute(stmt)
    await db.commit()
