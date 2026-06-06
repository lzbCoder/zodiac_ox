import time
import asyncio
import random
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import select, func, update, or_, and_
from loguru import logger
from database import async_session
from milvus_client import get_collection
from models.rag_eval_task import RagEvalTask
from models.rag_eval_question import RagEvalQuestion
from models.rag_eval_result import RagEvalResult
from models.chat_history import ChatHistory
from models.document_chunk import DocumentChunk
from services import rag_service
from services.ragas_eval_service import evaluate_ragas

# 内存取消标志 — 避免每题一次 db.refresh
_cancel_flags: dict[int, bool] = {}

# 并发控制
MAX_CONCURRENCY = 5


@dataclass
class EvalQuestion:
    """统一的问题结构，同时支持传统评测集和聊天抽样"""
    query: str
    standard_answer: str | None
    standard_chunk_ids: list[int] | None
    source_id: int  # RagEvalQuestion.id 或 ChatHistory.id


def request_cancel(task_id: int) -> None:
    _cancel_flags[task_id] = True


def _calculate_metrics(standard_chunk_ids: list[int] | None, retrieved_chunk_ids: list[int] | None, top_k: int) -> dict:
    std_set = set(standard_chunk_ids or [])
    ret_set = set(retrieved_chunk_ids or [])

    if not std_set:
        return {"recall": 0.0, "precision": 0.0, "hit": False, "rank": 0, "mrr": 0.0}

    hits = std_set & ret_set
    hit_count = len(hits)

    recall = hit_count / len(std_set)
    precision = hit_count / len(ret_set) if ret_set else 0.0
    hit = hit_count > 0

    rank = 0
    if hit and retrieved_chunk_ids:
        for i, cid in enumerate(retrieved_chunk_ids):
            if cid in std_set:
                rank = i + 1
                break

    mrr = 1.0 / rank if rank > 0 else 0.0
    return {"recall": recall, "precision": precision, "hit": hit, "rank": rank, "mrr": mrr}


async def _sample_chat_histories(db, task: RagEvalTask) -> list[EvalQuestion]:
    """从 chat_histories 按条件抽样，返回统一的问题结构列表。"""
    base = select(ChatHistory).where(
        ChatHistory.is_deleted == False,
        ChatHistory.user_query != "",
        ChatHistory.ai_answer != "",
    )
    if task.kb_id:
        base = base.where(ChatHistory.kb_id == task.kb_id)
    if task.sample_time_start:
        base = base.where(ChatHistory.created_at >= task.sample_time_start)
    if task.sample_time_end:
        base = base.where(ChatHistory.created_at <= task.sample_time_end)

    if task.sample_strategy == "latest":
        base = base.order_by(ChatHistory.created_at.desc()).limit(task.sample_count)
        rows = (await db.execute(base)).scalars().all()
    else:
        base = base.order_by(ChatHistory.id)
        rows = (await db.execute(base)).scalars().all()
        if len(rows) > task.sample_count:
            rows = random.sample(rows, task.sample_count)

    questions: list[EvalQuestion] = []
    for ch in rows:
        chunk_ids = await _map_ref_chunks_to_ids(db, ch.reference_chunks or [])
        questions.append(EvalQuestion(
            query=ch.user_query,
            standard_answer=ch.ai_answer,
            standard_chunk_ids=chunk_ids,
            source_id=ch.id,
        ))
    return questions


async def _map_ref_chunks_to_ids(db, reference_chunks: list[dict]) -> list[int]:
    """将 chat_histories.reference_chunks 中的 chunk_index 映射为 document_chunks.id。

    reference_chunks 结构: [{doc_id, filename, chunk_id: chunk_index, content, page_num, score}, ...]
    其中 chunk_id 实际存的是 chunk_index（文档内序号），需要与 doc_id 组合查询才能拿到 DB 主键。
    """
    if not reference_chunks:
        return []
    conditions = []
    for rc in reference_chunks:
        if "doc_id" in rc and "chunk_id" in rc:
            conditions.append(
                and_(DocumentChunk.doc_id == rc["doc_id"], DocumentChunk.chunk_index == rc["chunk_id"])
            )
    if not conditions:
        return []
    stmt = select(DocumentChunk.id).where(or_(*conditions))
    result = await db.execute(stmt)
    return [row[0] for row in result.all()]


async def run_evaluation(task_id: int):
    """后台运行评测。并行处理问题（Semaphore=5），批量提交，内存取消标志。"""
    start_time = time.perf_counter()
    _cancel_flags.pop(task_id, None)  # 清除旧标志

    async with async_session() as db:
        try:
            task = await db.get(RagEvalTask, task_id)
            if not task:
                logger.error(f"Eval task {task_id} not found")
                return

            if task.status == "cancelled":
                return

            task.status = "running"
            await db.commit()

            # 加载问题列表 — 根据任务类型分支
            if task.task_type == "chat_sample":
                questions = await _sample_chat_histories(db, task)
            else:
                q_stmt = select(RagEvalQuestion).where(
                    RagEvalQuestion.dataset_id == task.dataset_id,
                    RagEvalQuestion.is_deleted == False,
                )
                q_rows = (await db.execute(q_stmt)).scalars().all()
                questions = [
                    EvalQuestion(
                        query=q.query,
                        standard_answer=q.standard_answer,
                        standard_chunk_ids=q.standard_chunk_ids,
                        source_id=q.id,
                    )
                    for q in q_rows
                ]

            if not questions:
                task.status = "failed"
                await db.commit()
                logger.warning(f"Eval task {task_id}: no questions available")
                return

            total = len(questions)
            logger.info(
                f"Eval task {task_id}: starting with {total} questions, top_k={task.top_k}, "
                f"mode={task.retriever_mode}, ragas={task.enable_ragas}, concurrency={MAX_CONCURRENCY}, "
                f"type={task.task_type}"
            )

            collection = get_collection()
            collection.load()

            # 共享状态
            sem = asyncio.Semaphore(MAX_CONCURRENCY)
            completed = 0
            progress_lock = asyncio.Lock()

            async def process_one(q: EvalQuestion) -> tuple[dict, dict | None]:
                """处理单题。独立 DB 会话，通过信号量控制并发。"""
                nonlocal completed

                async with sem:
                    if _cancel_flags.get(task_id):
                        return {}, None

                    try:
                        t0 = time.perf_counter()
                        hits = await rag_service.search_vectors(
                            collection=collection,
                            query=q.query,
                            kb_id=task.kb_id,
                            search_mode=task.retriever_mode,
                            dense_top_k=task.top_k,
                            final_top_k=task.top_k,
                        )
                        retrieve_time = time.perf_counter() - t0

                        retrieved_chunk_ids = []
                        retrieved_doc_ids = []
                        for hit in hits:
                            if hit["chunk_id"] is not None:
                                retrieved_chunk_ids.append(int(hit["chunk_id"]))
                            if hit["doc_id"] is not None:
                                retrieved_doc_ids.append(int(hit["doc_id"]))

                        metrics = _calculate_metrics(
                            q.standard_chunk_ids,
                            retrieved_chunk_ids,
                            task.top_k,
                        )
                        metrics["retrieve_time"] = round(retrieve_time, 4)

                        # RAGAS 生成评测（需要独立 DB 会话获取 chunk 文本）
                        ragas = None
                        if task.enable_ragas:
                            try:
                                async with async_session() as ragas_db:
                                    ragas = await evaluate_ragas(
                                        db=ragas_db,
                                        query=q.query,
                                        ground_truth=q.standard_answer,
                                        chunk_ids=retrieved_chunk_ids,
                                        model=task.model_name or "qwen3-max",
                                        eval_model=task.eval_model or "qwen-turbo",
                                    )
                            except Exception as e:
                                logger.error(f"RAGAS eval failed for q{q.source_id}: {e}")

                        result_data = {
                            "task_id": task.id,
                            "qid": q.source_id,
                            "query": q.query,
                            "retrieved_chunk_ids": retrieved_chunk_ids,
                            "retrieved_doc_ids": retrieved_doc_ids,
                            "recall": round(metrics["recall"], 4),
                            "precision": round(metrics["precision"], 4),
                            "hit": metrics["hit"],
                            "rank": metrics["rank"],
                            "mrr": round(metrics["mrr"], 4),
                            "retrieve_time": round(retrieve_time, 4),
                            "answer_time": ragas["answer_time"] if ragas else 0,
                            "answer": ragas["answer"] if ragas else None,
                            "context_precision": ragas["context_precision"] if ragas else None,
                            "context_recall": ragas["context_recall"] if ragas else None,
                            "faithfulness": ragas["faithfulness"] if ragas else None,
                            "answer_relevancy": ragas["answer_relevancy"] if ragas else None,
                        }

                        # 提交结果到独立会话
                        async with async_session() as result_db:
                            result = RagEvalResult(**result_data)
                            result_db.add(result)
                            await result_db.commit()

                    except Exception as e:
                        logger.error(f"Eval task {task_id}: error on question q{q.source_id}: {e}")
                        metrics = {"recall": 0, "precision": 0, "hit": False, "rank": 0, "mrr": 0, "retrieve_time": 0}
                        result_data = None

                    # 更新进度（在独立会话中，避开 worker 的 result 会话）
                    async with progress_lock:
                        completed += 1
                        pct = int(completed / total * 100)
                        if pct % 10 == 0 or completed == total:
                            try:
                                async with async_session() as prog_db:
                                    await prog_db.execute(
                                        update(RagEvalTask)
                                        .where(RagEvalTask.id == task_id)
                                        .values(progress=pct)
                                    )
                                    await prog_db.commit()
                            except Exception:
                                pass

                    return metrics, result_data

            # 并行处理全部问题
            coros = [process_one(q) for q in questions]
            raw = await asyncio.gather(*coros)

            # 检查是否被取消
            if _cancel_flags.get(task_id):
                async with async_session() as final_db:
                    t = await final_db.get(RagEvalTask, task_id)
                    if t:
                        t.status = "cancelled"
                        t.finished_at = datetime.now()
                        await final_db.commit()
                logger.info(f"Eval task {task_id}: cancelled")
                return

            # 汇总检索指标
            all_metrics = [m for m, _ in raw if m and m.get("retrieve_time") is not None]
            n = len(all_metrics)
            if n > 0:
                task.recall = round(sum(m["recall"] for m in all_metrics) / n, 4)
                task.precision = round(sum(m["precision"] for m in all_metrics) / n, 4)
                task.hit_rate = round(sum(1 for m in all_metrics if m["hit"]) / n, 4)
                task.mrr = round(sum(m["mrr"] for m in all_metrics) / n, 4)
            else:
                task.recall = 0
                task.precision = 0
                task.hit_rate = 0
                task.mrr = 0

            # 汇总 RAGAS 平均分
            if task.enable_ragas:
                ragas_results = (await db.execute(
                    select(RagEvalResult).where(RagEvalResult.task_id == task.id)
                )).scalars().all()
                ragas_n = len([r for r in ragas_results if r.context_precision is not None])
                if ragas_n > 0:
                    task.context_precision = round(
                        sum(r.context_precision for r in ragas_results if r.context_precision is not None) / ragas_n, 4
                    )
                    task.context_recall = round(
                        sum(r.context_recall for r in ragas_results if r.context_recall is not None) / ragas_n, 4
                    )
                    task.faithfulness = round(
                        sum(r.faithfulness for r in ragas_results if r.faithfulness is not None) / ragas_n, 4
                    )
                    task.answer_relevancy = round(
                        sum(r.answer_relevancy for r in ragas_results if r.answer_relevancy is not None) / ragas_n, 4
                    )

            task.status = "completed"
            task.progress = 100
            task.cost_seconds = round(time.perf_counter() - start_time, 1)
            task.finished_at = datetime.now()
            await db.commit()

            logger.info(
                f"Eval task {task_id} completed: recall={task.recall:.4f}, precision={task.precision:.4f}, "
                f"hit_rate={task.hit_rate:.4f}, mrr={task.mrr:.4f}, cost={task.cost_seconds}s"
            )

        except asyncio.CancelledError:
            _cancel_flags.pop(task_id, None)
            task.status = "cancelled"
            task.finished_at = datetime.now()
            await db.commit()
            logger.info(f"Eval task {task_id}: cancelled via CancelledError")
        except Exception as e:
            _cancel_flags.pop(task_id, None)
            logger.error(f"Eval task {task_id} failed: {e}")
            try:
                task.status = "failed"
                await db.commit()
            except Exception:
                pass
