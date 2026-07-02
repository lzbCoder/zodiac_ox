"""
RAGAS 评测指标服务 —— 基于 ragas 库的四维生成质量评测
(context_precision / context_recall / faithfulness / answer_relevancy)

使用 ragas 0.4.x 最新 API：llm_factory() + metric.ascore()
（原 ragas.evaluate() / ragas.aevaluate() 已废弃）
"""
import time
import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from openai import AsyncOpenAI
from config import DASHSCOPE_API_KEY
from models.document_chunk import DocumentChunk
from loguru import logger

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy,
)
from services.ragas_embedding import DashScopeRagasEmbedding


async def fetch_chunk_contents(db: AsyncSession, chunk_ids: list[int]) -> list[dict]:
    """按 chunk_id 批量从 PG 获取 chunk 文本内容"""
    if not chunk_ids:
        return []
    unique_ids = list(set(chunk_ids))
    stmt = select(DocumentChunk).where(DocumentChunk.id.in_(unique_ids))
    result = await db.execute(stmt)
    chunks = result.scalars().all()
    return [{"chunk_id": c.id, "content": c.content, "doc_id": c.doc_id} for c in chunks]


async def generate_answer_for_eval(
    query: str, chunks: list[dict], model: str = "qwen3-max"
) -> str:
    """非流式调用 LLM 生成完整回答（供 RAGAS 评测用）"""
    if not chunks:
        context_text = "（无相关文档）"
    else:
        context_text = "\n\n---\n\n".join(
            f"[来源片段{c['chunk_id']}] {c['content']}" for c in chunks
        )

    system_prompt = (
        "你是一个知识库问答助手。请根据提供的文档片段回答用户问题。"
        "如果文档片段不足以回答问题，请如实说明。回答时请引用具体的来源。"
    )
    user_prompt = (
        f"文档片段：\n{context_text}\n\n"
        f"用户问题：{query}\n\n"
        f"请根据以上文档片段回答问题："
    )

    client = AsyncOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return f"[生成失败: {e}]"


async def evaluate_ragas(
    db: AsyncSession,
    query: str,
    ground_truth: str | None,
    chunk_ids: list[int],
    model: str = "qwen3-max",
    eval_model: str = "qwen3.6-flash",
) -> dict:
    """
    对单条问题执行 RAGAS 四维评测（使用 metric.ascore() 新 API）。
    model: 答案生成用模型（需要高质量）
    eval_model: RAGAS 评分用模型（轻量即可，默认 qwen3.6-flash）
    返回: {context_precision, context_recall, faithfulness, answer_relevancy, answer, answer_time}
    """
    # 1. 获取 chunk 文本
    chunks = await fetch_chunk_contents(db, chunk_ids)
    context_texts = [c["content"] for c in chunks]

    # 2. 生成非流式答案
    t0 = time.perf_counter()
    answer = await generate_answer_for_eval(query, chunks, model)
    answer_time = round(time.perf_counter() - t0, 4)

    # 3. 构建 ragas 评测器（使用 llm_factory 创建 InstructorLLM）
    openai_client = AsyncOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    evaluator_llm = llm_factory(eval_model, client=openai_client)
    embeddings = DashScopeRagasEmbedding()

    # 4. 构造指标实例（llm + embeddings 注入到构造器）
    cp_metric = ContextPrecision(llm=evaluator_llm, embeddings=embeddings)
    f_metric = Faithfulness(llm=evaluator_llm, embeddings=embeddings)
    ar_metric = AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings)

    # 5. 并行异步评分
    t_ragas_start = time.perf_counter()

    async def _safe_ascore(name: str, coro):
        """包装 ascore 调用，单指标失败不影响其他指标"""
        try:
            _t0 = time.perf_counter()
            result = await coro
            _elapsed = round(time.perf_counter() - _t0, 2)
            logger.info(f"  [{name}] score={float(result.value):.4f} in {_elapsed}s")
            return name, round(float(result.value), 4)
        except Exception as e:
            logger.error(f"{name}.ascore() failed for query '{query[:50]}': {e}")
            return name, 0.0

    tasks = [
        _safe_ascore("context_precision", cp_metric.ascore(
            user_input=query,
            reference=ground_truth or "",
            retrieved_contexts=context_texts,
        )),
        _safe_ascore("faithfulness", f_metric.ascore(
            user_input=query,
            response=answer,
            retrieved_contexts=context_texts,
        )),
        _safe_ascore("answer_relevancy", ar_metric.ascore(
            user_input=query,
            response=answer,
        )),
    ]

    # ContextRecall 需要 ground_truth，无标注时跳过
    if ground_truth:
        cr_metric = ContextRecall(llm=evaluator_llm, embeddings=embeddings)
        tasks.insert(1, _safe_ascore("context_recall", cr_metric.ascore(
            user_input=query,
            reference=ground_truth,
            retrieved_contexts=context_texts,
        )))

    scores = dict(await asyncio.gather(*tasks))

    t_ragas_total = round(time.perf_counter() - t_ragas_start, 2)
    logger.info(
        f"RAGAS q='{query[:40]}' gen_model={model} eval_model={eval_model} "
        f"answer_time={answer_time}s ragas_metrics={t_ragas_total}s total={round(answer_time + t_ragas_total, 2)}s"
    )

    return {
        "context_precision": scores.get("context_precision", 0.0),
        "context_recall": scores.get("context_recall", 0.0),
        "faithfulness": scores.get("faithfulness", 0.0),
        "answer_relevancy": scores.get("answer_relevancy", 0.0),
        "answer": answer,
        "answer_time": answer_time,
    }


def evaluate_ragas_sync(
    query: str,
    ground_truth: str | None,
    context_texts: list[str],
    model: str = "qwen3-max",
    eval_model: str = "qwen3.6-flash",
) -> dict:
    """sync 版 RAGAS 评测——专为 asyncio.to_thread 设计，避免事件循环冲突。

    与 evaluate_ragas 功能相同，但：
    - 使用 sync OpenAI 生成回答（直接阻塞，不出主事件循环）
    - 使用 AsyncOpenAI + asyncio.run() 执行 ragas ascore()，走正版 async 路径
    - 不依赖 DB 会话，context_texts 由调用方传入
    """
    from openai import OpenAI as SyncOpenAI

    # 1. 生成回答（sync，跑在线程中 = 自然阻塞，不伤事件循环）
    t0 = time.perf_counter()
    context_text = "\n\n---\n\n".join(
        f"[来源片段{i+1}] {c}" for i, c in enumerate(context_texts)
    ) if context_texts else "（无相关文档）"
    system_prompt = (
        "你是一个知识库问答助手。请根据提供的文档片段回答用户问题。"
        "如果文档片段不足以回答问题，请如实说明。回答时请引用具体的来源。"
    )
    user_prompt = (
        f"文档片段：\n{context_text}\n\n"
        f"用户问题：{query}\n\n"
        f"请根据以上文档片段回答问题："
    )
    sync_client = SyncOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        resp = sync_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        answer = resp.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Answer generation failed (sync): {e}")
        answer = f"[生成失败: {e}]"
    answer_time = round(time.perf_counter() - t0, 4)

    # 2. 构建 RAGAS 评测器（使用 AsyncOpenAI，满足 ascore() 的 async 要求）
    async_client = AsyncOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        llm = llm_factory(eval_model, client=async_client, max_tokens=4096)
        embeddings = DashScopeRagasEmbedding()

        cp_metric = ContextPrecision(llm=llm, embeddings=embeddings)
        f_metric = Faithfulness(llm=llm, embeddings=embeddings)
        ar_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)

        def _safe_ascore(name: str, metric, **kwargs):
            """在线程中执行 asyncio.run(metric.ascore(...))，走 ragas async 路径"""
            try:
                _t0 = time.perf_counter()
                result = asyncio.run(metric.ascore(**kwargs))
                _elapsed = round(time.perf_counter() - _t0, 2)
                logger.info(f"  [{name}] score={float(result):.4f} in {_elapsed}s")
                return name, round(float(result), 4)
            except Exception as e:
                logger.error(f"{name}.ascore() failed for query '{query[:50]}': {e}")
                return name, 0.0

        scores = dict([
            _safe_ascore("context_precision", cp_metric,
                         user_input=query, reference=ground_truth or "",
                         retrieved_contexts=context_texts),
            _safe_ascore("faithfulness", f_metric,
                         user_input=query, response=answer,
                         retrieved_contexts=context_texts),
            _safe_ascore("answer_relevancy", ar_metric,
                         user_input=query, response=answer),
        ])
        if ground_truth:
            cr_metric = ContextRecall(llm=llm, embeddings=embeddings)
            scores["context_recall"] = _safe_ascore(
                "context_recall", cr_metric,
                user_input=query, reference=ground_truth,
                retrieved_contexts=context_texts,
            )[1]

        t_ragas_total = round(time.perf_counter() - t0 - answer_time, 2)
        logger.info(
            f"RAGAS sync q='{query[:40]}' gen_model={model} eval_model={eval_model} "
            f"answer_time={answer_time}s ragas_metrics={t_ragas_total}s"
        )

        return {
            "context_precision": scores.get("context_precision", 0.0),
            "context_recall": scores.get("context_recall", 0.0),
            "faithfulness": scores.get("faithfulness", 0.0),
            "answer_relevancy": scores.get("answer_relevancy", 0.0),
            "answer": answer,
            "answer_time": answer_time,
        }
    finally:
        # 显式关闭 AsyncOpenAI 底层的 httpx 连接池，防止 asyncio.run() 关闭事件循环后
        # httpx 异步清理任务抛 "Event loop is closed" RuntimeError
        try:
            asyncio.run(async_client.close())
        except RuntimeError:
            pass
