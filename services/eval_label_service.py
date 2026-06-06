import csv, io
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
from models.rag_eval_label_task import RagEvalLabelTask
from models.rag_eval_label_detail import RagEvalLabelDetail
from models.document_chunk import DocumentChunk
from models.document import Document
from models.knowledge_base import KnowledgeBase
from milvus_client import get_collection
from services.rag_service import search_vectors


# ── 任务 CRUD ────────────────────────────────────────────

async def create_label_task(db: AsyncSession, name: str, kb_id: int, queries: list[dict], top_k: int = 5, description: str | None = None, created_by: str | None = None) -> RagEvalLabelTask:
    task = RagEvalLabelTask(
        name=name,
        kb_id=kb_id,
        top_k=top_k,
        description=description,
        created_by=created_by,
    )
    db.add(task)
    await db.flush()

    for q in queries:
        db.add(RagEvalLabelDetail(
            task_id=task.id,
            query=q["query"],
            standard_answer=q.get("standard_answer"),
            status="unannotated",
        ))

    await db.commit()
    await db.refresh(task)
    task.total_details = len(queries)
    return task


async def list_label_tasks(db: AsyncSession, kb_id: int | None = None, page: int = 1, page_size: int = 20) -> tuple[list[RagEvalLabelTask], int]:
    base = select(RagEvalLabelTask)
    if kb_id is not None:
        base = base.where(RagEvalLabelTask.kb_id == kb_id)

    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = base.order_by(RagEvalLabelTask.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()

    # 为每个任务附加 total_details 计数
    task_ids = [t.id for t in rows]
    if task_ids:
        count_subq = (
            select(RagEvalLabelDetail.task_id, func.count().label("cnt"))
            .where(RagEvalLabelDetail.task_id.in_(task_ids))
            .group_by(RagEvalLabelDetail.task_id)
            .subquery()
        )
        counts = {r.task_id: r.cnt for r in (await db.execute(select(count_subq))).all()}
    else:
        counts = {}

    for t in rows:
        t.total_details = counts.get(t.id, 0)

    return list(rows), total


async def get_label_task(db: AsyncSession, task_id: int) -> RagEvalLabelTask | None:
    task = await db.get(RagEvalLabelTask, task_id)
    if task:
        cnt_stmt = select(func.count()).where(RagEvalLabelDetail.task_id == task_id)
        task.total_details = (await db.execute(cnt_stmt)).scalar() or 0
    return task


async def update_label_task(db: AsyncSession, task_id: int, name: str | None = None, description: str | None = None) -> RagEvalLabelTask | None:
    task = await db.get(RagEvalLabelTask, task_id)
    if not task:
        return None
    if name is not None:
        task.name = name
    if description is not None:
        task.description = description
    task.updated_at = datetime.now()
    await db.commit()
    await db.refresh(task)
    cnt_stmt = select(func.count()).where(RagEvalLabelDetail.task_id == task_id)
    task.total_details = (await db.execute(cnt_stmt)).scalar() or 0
    return task


async def delete_label_task(db: AsyncSession, task_id: int) -> bool:
    task = await db.get(RagEvalLabelTask, task_id)
    if not task:
        return False
    await db.delete(task)
    await db.commit()
    return True


# ── 详情 ─────────────────────────────────────────────────

async def list_label_details(db: AsyncSession, task_id: int, status: str | None = None, keyword: str | None = None, page: int = 1, page_size: int = 50) -> tuple[list[RagEvalLabelDetail], int]:
    base = select(RagEvalLabelDetail).where(RagEvalLabelDetail.task_id == task_id)
    if status and status != "all":
        base = base.where(RagEvalLabelDetail.status == status)
    if keyword:
        base = base.where(RagEvalLabelDetail.query.ilike(f"%{keyword}%"))

    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = base.order_by(RagEvalLabelDetail.id).offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()
    return list(rows), total


async def get_label_detail(db: AsyncSession, detail_id: int) -> RagEvalLabelDetail | None:
    return await db.get(RagEvalLabelDetail, detail_id)


# ── 核心：候选 Chunk 检索 ───────────────────────────────

async def get_candidate_chunks(db: AsyncSession, detail_id: int) -> dict | None:
    detail = await db.get(RagEvalLabelDetail, detail_id)
    if not detail:
        return None

    task = await db.get(RagEvalLabelTask, detail.task_id)
    if not task:
        return None

    try:
        collection = get_collection()
        collection.load()
        hits = await search_vectors(collection, detail.query, task.kb_id,
                                    dense_top_k=task.top_k, final_top_k=task.top_k)
    except Exception as e:
        logger.error(f"Label chunk retrieval failed for detail {detail_id}: {e}")
        return {
            "detail_id": detail_id,
            "query": detail.query,
            "standard_answer": detail.standard_answer,
            "chunks": [],
        }

    if not hits:
        return {
            "detail_id": detail_id,
            "query": detail.query,
            "standard_answer": detail.standard_answer,
            "chunks": [],
        }

    # 按主键批量从 PostgreSQL 获取 chunk 内容（chunk_id 存储的是 DocumentChunk.id）
    chunk_ids = []
    for h in hits:
        c_idx = h.get("chunk_id")
        if c_idx is not None:
            chunk_ids.append(int(c_idx))

    if chunk_ids:
        chunk_stmt = select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
        chunks = (await db.execute(chunk_stmt)).scalars().all()
    else:
        chunks = []
    chunk_map: dict[int, DocumentChunk] = {c.id: c for c in chunks}

    # 批量获取文档文件名
    doc_ids = list({c.doc_id for c in chunks})
    doc_stmt = select(Document).where(Document.id.in_(doc_ids))
    docs = (await db.execute(doc_stmt)).scalars().all()
    doc_map = {d.id: d for d in docs}

    # 组装增强输出，保持 Milvus 排序顺序
    enriched = []
    for h in hits:
        c_idx = h.get("chunk_id")
        if c_idx is None:
            continue
        c = chunk_map.get(int(c_idx))
        if not c:
            continue
        d = doc_map.get(c.doc_id)
        enriched.append({
            "chunk_id": c.id,
            "doc_id": c.doc_id,
            "kb_id": h["kb_id"],
            "score": round(h["score"], 4),
            "content": c.content,
            "doc_name": d.filename if d else "未知文档",
            "page_num": c.page_num,
        })

    return {
        "detail_id": detail_id,
        "query": detail.query,
        "standard_answer": detail.standard_answer,
        "chunks": enriched,
    }


# ── 保存标注 ────────────────────────────────────────────

async def save_detail_annotation(db: AsyncSession, detail_id: int, standard_chunk_ids: list[int], annotated_by: str | None = None) -> RagEvalLabelDetail | None:
    detail = await db.get(RagEvalLabelDetail, detail_id)
    if not detail:
        return None

    detail.standard_chunk_ids = standard_chunk_ids

    # 从 DocumentChunk 主键推导 standard_doc_ids
    if standard_chunk_ids:
        stmt = select(DocumentChunk.doc_id).where(
            DocumentChunk.id.in_(standard_chunk_ids),
        )
        results = (await db.execute(stmt)).scalars().all()
        detail.standard_doc_ids = list(dict.fromkeys(results))
    else:
        detail.standard_doc_ids = []

    detail.status = "annotated"
    detail.annotated_by = annotated_by
    detail.annotated_at = datetime.now()
    await db.commit()
    await db.refresh(detail)

    await _update_task_progress(db, detail.task_id)
    return detail


async def _update_task_progress(db: AsyncSession, task_id: int):
    total_stmt = select(func.count()).where(RagEvalLabelDetail.task_id == task_id)
    annotated_stmt = select(func.count()).where(
        RagEvalLabelDetail.task_id == task_id,
        RagEvalLabelDetail.status == "annotated",
    )
    total = (await db.execute(total_stmt)).scalar() or 0
    annotated = (await db.execute(annotated_stmt)).scalar() or 0
    progress = int(annotated / total * 100) if total > 0 else 0
    status = "completed" if total > 0 and annotated >= total else "in_progress"

    task = await db.get(RagEvalLabelTask, task_id)
    if task:
        task.progress = progress
        task.status = status
        await db.commit()


async def batch_save_annotations(db: AsyncSession, annotations: list[dict], annotated_by: str | None = None) -> int:
    saved = 0
    for ann in annotations:
        detail_id = ann.get("detail_id")
        chunk_ids = ann.get("standard_chunk_ids", [])
        if detail_id is not None:
            detail = await db.get(RagEvalLabelDetail, detail_id)
            if detail:
                detail.standard_chunk_ids = chunk_ids
                if chunk_ids:
                    stmt = select(DocumentChunk.doc_id).where(
                        DocumentChunk.id.in_(chunk_ids),
                    )
                    results = (await db.execute(stmt)).scalars().all()
                    detail.standard_doc_ids = list(dict.fromkeys(results))
                else:
                    detail.standard_doc_ids = []
                detail.status = "annotated"
                detail.annotated_by = annotated_by
                detail.annotated_at = datetime.now()
                saved += 1

    if saved > 0:
        await db.commit()
        # 更新第一个标注所属任务的进度
        first_task_id = annotations[0].get("task_id") if annotations else None
        if not first_task_id:
            detail = await db.get(RagEvalLabelDetail, annotations[0]["detail_id"])
            if detail:
                first_task_id = detail.task_id
        if first_task_id:
            await _update_task_progress(db, first_task_id)

    return saved


# ── 导出 ────────────────────────────────────────────────

async def export_label_annotations(db: AsyncSession, task_id: int, format: str = "json") -> dict | str | bytes:
    stmt = select(RagEvalLabelDetail).where(
        RagEvalLabelDetail.task_id == task_id,
        RagEvalLabelDetail.status == "annotated",
    ).order_by(RagEvalLabelDetail.id)
    details = (await db.execute(stmt)).scalars().all()

    if format == "json":
        return [{
            "query": d.query,
            "standard_answer": d.standard_answer or "",
            "standard_chunk_ids": d.standard_chunk_ids or [],
            "standard_doc_ids": d.standard_doc_ids or [],
            "difficulty": "medium",
        } for d in details]

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["query", "standard_answer", "standard_chunk_ids", "standard_doc_ids", "difficulty"])
        for d in details:
            chunk_ids = ",".join(str(x) for x in (d.standard_chunk_ids or []))
            doc_ids = ",".join(str(x) for x in (d.standard_doc_ids or []))
            writer.writerow([d.query, d.standard_answer or "", chunk_ids, doc_ids, "medium"])
        return output.getvalue()

    if format == "xlsx":
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "评测集"
            ws.append(["query", "standard_answer", "standard_chunk_ids", "standard_doc_ids", "difficulty"])
            for d in details:
                ws.append([
                    d.query,
                    d.standard_answer or "",
                    ",".join(str(x) for x in (d.standard_chunk_ids or [])),
                    ",".join(str(x) for x in (d.standard_doc_ids or [])),
                    "medium",
                ])
            buf = io.BytesIO()
            wb.save(buf)
            buf.seek(0)
            return buf.getvalue()
        except ImportError:
            raise ImportError("openpyxl is required for Excel export")

    raise ValueError(f"Unsupported export format: {format}")
