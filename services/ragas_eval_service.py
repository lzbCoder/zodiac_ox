"""
RAGAS 评测指标服务 —— 基于 ragas 库的四维生成质量评测
(context_precision / context_recall / faithfulness / answer_relevancy)

使用 ragas 0.4.x 最新 API：llm_factory() + metric.ascore()
（原 ragas.evaluate() / ragas.aevaluate() 已废弃）

对外唯一入口：evaluate_ragas_sync() —— 在 asyncio.to_thread 的线程中创建
独立事件循环，一次 asyncio.run() 并发完成答案生成 + 所有指标评测。
"""
import time
import asyncio
from openai import AsyncOpenAI
from config import DASHSCOPE_API_KEY
from loguru import logger

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy,
)
from services.ragas_embedding import DashScopeRagasEmbedding


def evaluate_ragas_sync(
    query: str,
    ground_truth: str | None,
    context_texts: list[str],
    model: str = "qwen3-max",
    eval_model: str = "qwen3.6-flash",
) -> dict:
    """sync 版 RAGAS 评测——专为 asyncio.to_thread 设计，避免事件循环冲突。

    在线程中创建独立事件循环（asyncio.run），所有异步工作（答案生成 +
    RAGAS 三/四维指标）在该循环中并发完成，彻底消除「Event loop is closed」
    噪音，并将指标评测耗时从 ~160s 降到 ~60s。
    """
    t0 = time.perf_counter()

    async def _run_all() -> dict:
        # ── 1. 构建提示词 ──
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

        # ── 2. 答案生成 + RAGAS 指标评测（同一 AsyncOpenAI 客户端，同一事件循环）──
        async with AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ) as client:
            # 2a. 生成答案
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
                answer = resp.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"Answer generation failed (sync): {e}")
                answer = f"[生成失败: {e}]"
            answer_time = round(time.perf_counter() - t0, 4)

            # 2b. 构建指标实例
            llm = llm_factory(eval_model, client=client, max_tokens=4096)
            embeddings = DashScopeRagasEmbedding()

            cp_metric = ContextPrecision(llm=llm, embeddings=embeddings)
            f_metric = Faithfulness(llm=llm, embeddings=embeddings)
            ar_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)

            # 2c. 并发执行所有指标（一个事件循环、一次 gather）
            async def _safe_ascore(name: str, coro):
                try:
                    _t1 = time.perf_counter()
                    result = await coro
                    _elapsed = round(time.perf_counter() - _t1, 2)
                    logger.info(f"  [{name}] score={float(result):.4f} in {_elapsed}s")
                    return name, round(float(result), 4)
                except Exception as e:
                    logger.error(f"{name}.ascore() failed for query '{query[:50]}': {e}")
                    return name, 0.0

            t_ragas_start = time.perf_counter()
            coros = [
                _safe_ascore("context_precision", cp_metric.ascore(
                    user_input=query, reference=ground_truth or "",
                    retrieved_contexts=context_texts,
                )),
                _safe_ascore("faithfulness", f_metric.ascore(
                    user_input=query, response=answer,
                    retrieved_contexts=context_texts,
                )),
                _safe_ascore("answer_relevancy", ar_metric.ascore(
                    user_input=query, response=answer,
                )),
            ]
            if ground_truth:
                cr_metric = ContextRecall(llm=llm, embeddings=embeddings)
                coros.insert(1, _safe_ascore("context_recall", cr_metric.ascore(
                    user_input=query, reference=ground_truth,
                    retrieved_contexts=context_texts,
                )))
            scores = dict(await asyncio.gather(*coros))

            t_ragas_total = round(time.perf_counter() - t_ragas_start, 2)
            logger.info(
                f"RAGAS sync q='{query[:40]}' gen_model={model} eval_model={eval_model} "
                f"answer_time={answer_time}s ragas_metrics={t_ragas_total}s "
                f"total={round(answer_time + t_ragas_total, 2)}s"
            )

            return {
                "context_precision": scores.get("context_precision", 0.0),
                "context_recall": scores.get("context_recall", 0.0),
                "faithfulness": scores.get("faithfulness", 0.0),
                "answer_relevancy": scores.get("answer_relevancy", 0.0),
                "answer": answer,
                "answer_time": answer_time,
            }

    return asyncio.run(_run_all())
