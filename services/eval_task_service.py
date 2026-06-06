from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from models.rag_eval_task import RagEvalTask
from models.rag_eval_result import RagEvalResult
from models.rag_eval_dataset import RagEvalDataset
from models.knowledge_base import KnowledgeBase


async def create_task(
    db: AsyncSession,
    name: str,
    dataset_id: int | None,
    kb_id: int,
    top_k: int = 5,
    retriever_mode: str = "normal",
    model_name: str | None = None,
    enable_ragas: bool = False,
    eval_model: str | None = None,
    task_type: str = "manual",
    sample_time_start: datetime | None = None,
    sample_time_end: datetime | None = None,
    sample_count: int = 10,
    sample_strategy: str = "random",
) -> RagEvalTask:
    task = RagEvalTask(
        name=name,
        task_type=task_type,
        dataset_id=dataset_id,
        kb_id=kb_id,
        top_k=top_k,
        retriever_mode=retriever_mode,
        model_name=model_name,
        enable_ragas=enable_ragas,
        eval_model=eval_model,
        sample_time_start=sample_time_start,
        sample_time_end=sample_time_end,
        sample_count=sample_count,
        sample_strategy=sample_strategy,
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    # Attach names for response
    if dataset_id:
        ds = await db.get(RagEvalDataset, dataset_id)
        task.dataset_name = ds.name if ds else ""
    else:
        task.dataset_name = "聊天抽样"
    kb = await db.get(KnowledgeBase, kb_id)
    task.kb_name = kb.name if kb else ""

    return task


def _attach_names(task: RagEvalTask, ds_name: str, kb_name: str) -> RagEvalTask:
    task.dataset_name = ds_name
    task.kb_name = kb_name
    return task


async def list_tasks(db: AsyncSession, kb_id: int | None = None, page: int = 1, page_size: int = 20) -> tuple[list[RagEvalTask], int]:
    base = select(RagEvalTask)
    if kb_id is not None:
        base = base.where(RagEvalTask.kb_id == kb_id)

    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = (
        select(RagEvalTask, RagEvalDataset.name, KnowledgeBase.name)
        .join(RagEvalDataset, RagEvalTask.dataset_id == RagEvalDataset.id, isouter=True)
        .join(KnowledgeBase, RagEvalTask.kb_id == KnowledgeBase.id)
    )
    if kb_id is not None:
        stmt = stmt.where(RagEvalTask.kb_id == kb_id)
    stmt = stmt.order_by(RagEvalTask.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).all()
    return [_attach_names(t, ds or "聊天抽样", kb) for t, ds, kb in rows], total


async def get_task(db: AsyncSession, task_id: int) -> RagEvalTask | None:
    stmt = (
        select(RagEvalTask, RagEvalDataset.name, KnowledgeBase.name)
        .join(RagEvalDataset, RagEvalTask.dataset_id == RagEvalDataset.id, isouter=True)
        .join(KnowledgeBase, RagEvalTask.kb_id == KnowledgeBase.id)
        .where(RagEvalTask.id == task_id)
    )
    row = (await db.execute(stmt)).first()
    if row:
        return _attach_names(row[0], row[1] or "聊天抽样", row[2])
    return None


async def cancel_task(db: AsyncSession, task_id: int) -> bool:
    task = await db.get(RagEvalTask, task_id)
    if not task or task.status not in ("pending", "running"):
        return False
    task.status = "cancelled"
    task.finished_at = datetime.now()
    await db.commit()
    return True


async def delete_task(db: AsyncSession, task_id: int) -> bool:
    task = await db.get(RagEvalTask, task_id)
    if not task:
        return False
    await db.delete(task)
    await db.commit()
    return True


async def save_result(db: AsyncSession, result_data: dict) -> RagEvalResult:
    r = RagEvalResult(**result_data)
    db.add(r)
    return r


async def get_results(db: AsyncSession, task_id: int, hit_filter: str | None = None) -> list[RagEvalResult]:
    stmt = select(RagEvalResult).where(RagEvalResult.task_id == task_id)
    if hit_filter == "hit":
        stmt = stmt.where(RagEvalResult.hit == True)
    elif hit_filter == "miss":
        stmt = stmt.where(RagEvalResult.hit == False)
    stmt = stmt.order_by(RagEvalResult.id)
    rows = (await db.execute(stmt)).scalars().all()
    return list(rows)


async def get_completed_tasks(db: AsyncSession, kb_id: int | None = None, page: int = 1, page_size: int = 20) -> tuple[list[RagEvalTask], int]:
    """列出已完成的任务，用于报告索引页。"""
    base = select(RagEvalTask).where(RagEvalTask.status == "completed")
    if kb_id is not None:
        base = base.where(RagEvalTask.kb_id == kb_id)

    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = (
        select(RagEvalTask, RagEvalDataset.name, KnowledgeBase.name)
        .join(RagEvalDataset, RagEvalTask.dataset_id == RagEvalDataset.id, isouter=True)
        .join(KnowledgeBase, RagEvalTask.kb_id == KnowledgeBase.id)
        .where(RagEvalTask.status == "completed")
    )
    if kb_id is not None:
        stmt = stmt.where(RagEvalTask.kb_id == kb_id)
    stmt = stmt.order_by(RagEvalTask.finished_at.desc()).offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).all()
    return [_attach_names(t, ds or "聊天抽样", kb) for t, ds, kb in rows], total
