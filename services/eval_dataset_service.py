from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from models.rag_eval_dataset import RagEvalDataset
from models.rag_eval_question import RagEvalQuestion


async def create_dataset(db: AsyncSession, kb_id: int, name: str, description: str | None = None, created_by: str | None = None) -> RagEvalDataset:
    ds = RagEvalDataset(kb_id=kb_id, name=name, description=description, created_by=created_by)
    db.add(ds)
    await db.commit()
    await db.refresh(ds)
    return ds


async def list_datasets(db: AsyncSession, kb_id: int | None = None, page: int = 1, page_size: int = 20) -> tuple[list[RagEvalDataset], int]:
    base = select(RagEvalDataset).where(RagEvalDataset.is_deleted == False)
    if kb_id is not None:
        base = base.where(RagEvalDataset.kb_id == kb_id)

    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = base.order_by(RagEvalDataset.updated_at.desc()).offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()
    return list(rows), total


async def get_dataset(db: AsyncSession, dataset_id: int) -> RagEvalDataset | None:
    stmt = select(RagEvalDataset).where(RagEvalDataset.id == dataset_id, RagEvalDataset.is_deleted == False)
    return (await db.execute(stmt)).scalar_one_or_none()


async def update_dataset(db: AsyncSession, dataset_id: int, name: str | None = None, description: str | None = None) -> RagEvalDataset | None:
    ds = await get_dataset(db, dataset_id)
    if not ds:
        return None
    if name is not None:
        ds.name = name
    if description is not None:
        ds.description = description
    await db.commit()
    await db.refresh(ds)
    return ds


async def delete_dataset(db: AsyncSession, dataset_id: int) -> bool:
    ds = await get_dataset(db, dataset_id)
    if not ds:
        return False
    ds.is_deleted = True
    await db.commit()
    return True


async def import_questions_from_rows(db: AsyncSession, dataset_id: int, kb_id: int, rows: list[dict]) -> int:
    count = 0
    for row in rows:
        q = RagEvalQuestion(
            dataset_id=dataset_id,
            kb_id=kb_id,
            query=row["query"],
            standard_answer=row.get("standard_answer"),
            standard_doc_ids=row.get("standard_doc_ids"),
            standard_chunk_ids=row.get("standard_chunk_ids"),
            difficulty=row.get("difficulty", "medium"),
        )
        db.add(q)
        count += 1

    # Update the dataset's total_questions counter
    await db.execute(
        update(RagEvalDataset)
        .where(RagEvalDataset.id == dataset_id)
        .values(total_questions=RagEvalDataset.total_questions + count)
    )
    await db.commit()
    return count


async def list_questions(db: AsyncSession, dataset_id: int, page: int = 1, page_size: int = 100) -> tuple[list[RagEvalQuestion], int]:
    base = select(RagEvalQuestion).where(
        RagEvalQuestion.dataset_id == dataset_id,
        RagEvalQuestion.is_deleted == False,
    )
    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = base.order_by(RagEvalQuestion.id).offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()
    return list(rows), total


async def delete_question(db: AsyncSession, question_id: int) -> bool:
    stmt = select(RagEvalQuestion).where(RagEvalQuestion.id == question_id, RagEvalQuestion.is_deleted == False)
    q = (await db.execute(stmt)).scalar_one_or_none()
    if not q:
        return False
    q.is_deleted = True
    await db.commit()
    return True
