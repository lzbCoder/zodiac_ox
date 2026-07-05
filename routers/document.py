from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
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
    file_category: str = Form(...),
    chunk_strategy: str = Form("default"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(100),
    split_separator: str = Form("\n\n"),
    language: str | None = Form(None),
    chunk_lines: int = Form(40),
    chunk_lines_overlap: int = Form(3),
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

        # 解析文档为 chunks（与预览共用同一套切分逻辑）
        chunks = await parsing_service.parse_document(
            file_path, chunk_size, chunk_overlap, split_separator,
            file_category=file_category,
            chunk_strategy=chunk_strategy,
            language=language,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
        )
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

    chunks = await document_service.get_chunks_by_doc(db, doc_id)

    # 从分片文本重建全文内容，确保每个分片的 start/end 位置在前端
    # 切片时完全匹配，不受 PDF 等文件提取非确定性的影响。
    parts: list[str] = []
    cumulative = 0
    corrected: list[ChunkPreview] = []
    for i, c in enumerate(chunks):
        # SQLAlchemy 异步 ORM 对 TEXT 列可能返回惰性代理对象，
        # 导致 len() 与完整序列化后的长度不一致，始终用 str() 强制
        # 转为普通 Python 字符串以保证字符数一致。
        chunk_text = str(c.content)
        clen = len(chunk_text)
        parts.append(chunk_text)
        corrected.append(ChunkPreview(
            chunk_index=c.chunk_index,
            content=chunk_text,
            page_num=c.page_num,
            start_pos=cumulative,
            end_pos=cumulative + clen,
        ))
        cumulative += clen
        if i < len(chunks) - 1:
            cumulative += 2  # 分片间 "\n\n" 分隔符
    content = "\n\n".join(parts)

    return DocumentPreview(
        filename=doc.filename,
        file_type=doc.file_type,
        content=content,
        chunks=corrected,
    )


_IMAGE_FILE_TYPES = {"jpg", "jpeg", "png", "gif", "webp", "bmp"}


@router.get("/{doc_id}/file")
async def get_document_file(doc_id: int, db: AsyncSession = Depends(get_db)):
    """返回文档的原始文件，图片可直接在 <img> 中引用。"""
    doc = await document_service.get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    if doc.file_type.lower() not in _IMAGE_FILE_TYPES:
        raise HTTPException(status_code=400, detail="仅支持图片文件预览")
    return FileResponse(doc.file_path)


@router.post("/preview-chunks-by-parser")
async def preview_chunks_by_parser(
    file: UploadFile = File(...),
    file_category: str = Form(...),
    chunk_strategy: str = Form("default"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(100),
    split_separator: str = Form("\n\n"),
    language: str | None = Form(None),
    chunk_lines: int = Form(40),
    chunk_lines_overlap: int = Form(3),
):
    """按文件类型分类 + 切片策略预览分片，与实际上传共用同一套切分逻辑。"""
    file_path, _, _ = await document_service.save_upload_file(file, 0)
    chunks = await parsing_service.parse_document(
        file_path, chunk_size, chunk_overlap, split_separator,
        file_category=file_category,
        chunk_strategy=chunk_strategy,
        language=language,
        chunk_lines=chunk_lines,
        chunk_lines_overlap=chunk_lines_overlap,
    )
    return {"chunks": [ChunkPreview(**c) for c in chunks]}


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
