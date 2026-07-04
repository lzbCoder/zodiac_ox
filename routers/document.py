from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from schemas.document import ChunkConfigInput, DocumentResponse, DocumentPreview, ChunkPreview
from services import document_service, parsing_service, embedding_service
from milvus_client import get_collection

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10M — 单文件业务限制

router = APIRouter(prefix="/api/documents", tags=["文档管理"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    kb_id: int = Form(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(100),
    split_separator: str = Form("\n\n"),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 保存文件
    file_path, file_type, file_size = await document_service.save_upload_file(file, kb_id)

    # 业务规则：单文件最大 10M
    if file_size > MAX_UPLOAD_SIZE:
        document_service.delete_local_file(file_path)
        raise HTTPException(status_code=400, detail="单文件最大支持 10M，当前文件超出限制")

    doc = await document_service.create_document(db, kb_id, file.filename, file_type, file_path, file_size)

    try:
        await document_service.update_document_status(db, doc.id, upload_status="processing")

        # 解析文档为 chunks
        chunks = await parsing_service.parse_document(file_path, chunk_size, chunk_overlap, split_separator)
        if not chunks:
            await document_service.update_document_status(db, doc.id, upload_status="failed")
            raise HTTPException(status_code=400, detail="无法解析文档内容")

        # 先保存 chunks 到 PG 以获取自增 ID
        chunk_objs = await document_service.save_chunks(db, kb_id, doc.id, chunks)
        for chunk, obj in zip(chunks, chunk_objs):
            chunk["id"] = obj.id

        # 嵌入并插入 Milvus（此时 chunk["id"] 为 PG 主键）
        texts = [c["content"] for c in chunks]
        embeddings = await embedding_service.embed_texts(texts)
        collection = get_collection()
        milvus_ids = await embedding_service.insert_vectors(collection, kb_id, doc.id, chunks, embeddings)

        # 更新 chunks 的 milvus_id
        for chunk, mid in zip(chunks, milvus_ids):
            chunk["milvus_id"] = mid
        await document_service.update_chunks_milvus_ids(db, [c["id"] for c in chunks], milvus_ids)
        await document_service.update_document_status(db, doc.id, upload_status="success", vector_status="completed", chunk_count=len(chunks))

    except Exception as e:
        await db.rollback()
        await document_service.update_document_status(db, doc.id, upload_status="failed")
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "上传成功", "doc_id": doc.id, "chunk_count": len(chunks)}


@router.get("")
async def list_documents(
    kb_id: int | None = Query(None),
    file_type: str | None = Query(None),
    filename: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    docs, total = await document_service.list_documents(db, kb_id, file_type, filename, page, page_size)
    return {"items": [DocumentResponse.model_validate(d) for d in docs], "total": total, "page": page, "page_size": page_size}


@router.get("/file-types")
async def list_file_types(db: AsyncSession = Depends(get_db)):
    """获取所有已上传文档的文件类型（去重）。"""
    types = await document_service.get_distinct_file_types(db)
    return types


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: int, db: AsyncSession = Depends(get_db)):
    doc = await document_service.get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return doc


@router.get("/{doc_id}/preview")
async def preview_document(doc_id: int, db: AsyncSession = Depends(get_db)):
    doc = await document_service.get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    content = await parsing_service.read_file_content(doc.file_path)
    chunks = await document_service.get_chunks_by_doc(db, doc_id)
    return DocumentPreview(
        filename=doc.filename,
        file_type=doc.file_type,
        content=content,
        chunks=[ChunkPreview(chunk_index=c.chunk_index, content=c.content, page_num=c.page_num, start_pos=c.start_pos, end_pos=c.end_pos) for c in chunks],
    )


@router.post("/preview-chunks")
async def preview_chunks(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(100),
    split_separator: str = Form("\n\n"),
):
    file_path, _, _ = await document_service.save_upload_file(file, 0)
    chunks = await parsing_service.preview_chunks(file_path, chunk_size, chunk_overlap, split_separator)
    return {"chunks": [ChunkPreview(chunk_index=c["chunk_index"], content=c["content"], page_num=c["page_num"], start_pos=c.get("start_pos", 0), end_pos=c.get("end_pos", 0)) for c in chunks]}


@router.delete("/{doc_id}")
async def delete_document(doc_id: int, db: AsyncSession = Depends(get_db)):
    doc = await document_service.get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 从 Milvus 删除向量
    collection = get_collection()
    embedding_service.delete_vectors_by_doc(collection, doc_id)

    # 删除本地文件
    document_service.delete_local_file(doc.file_path)

    # PG 中软删除
    await document_service.delete_document(db, doc_id)

    return {"message": "已删除"}


@router.get("/chunk-config/{kb_id}")
async def get_chunk_config(kb_id: int, db: AsyncSession = Depends(get_db)):
    config = await document_service.get_or_create_chunk_config(db, kb_id)
    return {
        "kb_id": config.kb_id,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "split_separator": config.split_separator,
    }


@router.put("/chunk-config/{kb_id}")
async def update_chunk_config(kb_id: int, data: ChunkConfigInput, db: AsyncSession = Depends(get_db)):
    config = await document_service.update_chunk_config(db, kb_id, data.chunk_size, data.chunk_overlap, data.split_separator)
    return {
        "kb_id": config.kb_id,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "split_separator": config.split_separator,
    }
