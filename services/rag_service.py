from typing import AsyncGenerator
import asyncio
from pymilvus import Collection, AnnSearchRequest, RRFRanker
from dashscope import TextEmbedding
from openai import AsyncOpenAI
from opentelemetry import trace, context as otel_context
from config import DASHSCOPE_API_KEY, EMBEDDING_DIM
from cache.model_config_cache import get_embedding_model
from services.sparse_embedding import JiebaSparseEmbedding

_sparse_encoder = JiebaSparseEmbedding()

# Markdown 排版指令：追加到 system prompt，保证模型以结构化 Markdown 输出
MARKDOWN_DIRECTIVE = (
    "\n\n【输出格式要求】请始终使用 Markdown 格式组织回答："
    "用 `##`/`###` 标题分层，用 `-` 或有序列表罗列要点，"
    "关键术语用 `**加粗**`，代码、命令或字段名用反引号或 ``` 代码块包裹，"
    "对比类信息使用表格，确保排版清晰、层次分明。"
)

# 引用来源指令：要求模型在回答末尾单独列出来源，正文中不插入来源标识
CITATION_DIRECTIVE = (
    "\n\n【引用来源要求】正文中不要插入来源标识，也不要在开头说明来源；"
    "请在回答全部结束后另起一行，先输出一行 `---` 分隔线，"
    "再以 `**参考来源：**` 开头，用列表逐条列出你实际引用到的文档名称（重复的只列一次）。"
    "若回答未引用任何文档，则不输出该来源小节。"
)

async def search_vectors(
    collection: Collection,
    query: str,
    kb_id: int,
    search_mode: str = "normal",
    dense_top_k: int = 5,
    sparse_top_k: int = 5,
    final_top_k: int = 5,
) -> list[dict]:
    with trace.get_tracer(__name__).start_as_current_span("milvus_retrieval") as span:
        span.set_attribute("kb_id", kb_id)
        span.set_attribute("search_mode", search_mode)
        span.set_attribute("dense_top_k", dense_top_k)
        span.set_attribute("sparse_top_k", sparse_top_k)
        span.set_attribute("final_top_k", final_top_k)

        resp = await asyncio.to_thread(
            TextEmbedding.call,
            api_key=DASHSCOPE_API_KEY,
            model=get_embedding_model(),
            input=query[:2048],
            dimensions=EMBEDDING_DIM,
        )
        if resp.status_code != 200:
            span.set_attribute("error", True)
            return []
        query_vector = resp.output["embeddings"][0]["embedding"]

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        expr = f"kb_id == {kb_id}"

        if search_mode == "hybrid":
            sparse_vec = _sparse_encoder.encode_queries([query])[0]
            dense_req = AnnSearchRequest(
                data=[query_vector],
                anns_field="dense_vector",
                param=search_params,
                limit=dense_top_k,
                expr=expr,
            )
            sparse_req = AnnSearchRequest(
                data=[sparse_vec],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=sparse_top_k,
                expr=expr,
            )
            results = collection.hybrid_search(
                reqs=[dense_req, sparse_req],
                rerank=RRFRanker(k=60),
                limit=final_top_k,
                output_fields=["kb_id", "doc_id", "chunk_id"],
            )
        else:
            results = collection.search(
                data=[query_vector],
                anns_field="dense_vector",
                param=search_params,
                limit=max(dense_top_k, final_top_k),
                expr=expr,
                output_fields=["kb_id", "doc_id", "chunk_id"],
            )

        hits = []
        for hit in results[0]:
            raw_score = hit.score
            # 普通搜索 (COSINE)：hit.score 就是余弦相似度 [-1, 1]，越高越相似
            # 混合搜索 (RRF)：hit.score 是 RRF 排名融合分，已"越高越好"
            if search_mode != "hybrid":
                score = max(0.0, raw_score)  # 负相似度（不相关）截断为 0
            else:
                score = raw_score
            hits.append({
                "id": hit.id,
                "doc_id": hit.entity.get("doc_id"),
                "chunk_id": hit.entity.get("chunk_id"),
                "kb_id": hit.entity.get("kb_id"),
                "score": score,
            })
        span.set_attribute("retrieved_count", len(hits))
        return hits[:final_top_k]


async def generate_answer_stream(
    query: str,
    context_chunks: list[dict],
    model: str = "qwen3-max",
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    _result: dict | None = None,
) -> AsyncGenerator[str, None]:
    """生成流式回答。

    可通过 ``_result`` 传入可变 dict，在迭代结束后获取额外信息：
    - ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``
    - ``system_prompt`` / ``user_prompt``（最终构建后的完整文本）
    """
    span = trace.get_tracer(__name__).start_span("llm_generate")
    span.set_attribute("model", model)
    span.set_attribute("context_chunks", len(context_chunks))
    span.set_attribute("query_length", len(query))
    ctx = otel_context.attach(trace.set_span_in_context(span))

    try:
        context_text = "\n\n---\n\n".join(
            f"[来源：{c.get('filename') or '未知文档'}] {c['content']}" for c in context_chunks
        )

        if system_prompt is None:
            system_prompt = (
                "你是一个知识库问答助手。请根据提供的文档片段回答用户问题。"
                "如果文档片段不足以回答问题，请如实说明。回答时请引用具体的来源。"
            )

        # 始终追加 Markdown 排版要求 + 引用来源要求（来源统一列在回答末尾）
        system_prompt = system_prompt + MARKDOWN_DIRECTIVE + CITATION_DIRECTIVE

        if user_prompt_template is None:
            user_prompt = f"文档片段：\n{context_text}\n\n用户问题：{query}\n\n请根据以上文档片段回答问题："
        else:
            user_prompt = user_prompt_template.replace("{context}", context_text).replace("{query}", query)

        # 回传最终 prompts（供保存入库）
        if _result is not None:
            _result["system_prompt"] = system_prompt
            _result["user_prompt"] = user_prompt

        client = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )

        token_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        async for chunk in stream:
            # 部分 provider 在最后一个 chunk 中返回 usage（choices 为空）
            if chunk.choices and chunk.choices[0].delta.content:
                token_count += 1
                yield chunk.choices[0].delta.content
            # OpenAI / DashScope 兼容：最后一条 usage chunk
            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
                total_tokens = chunk.usage.total_tokens or 0

        # 回传 token 用量
        if _result is not None:
            _result["prompt_tokens"] = prompt_tokens
            _result["completion_tokens"] = completion_tokens if completion_tokens else token_count
            _result["total_tokens"] = total_tokens if total_tokens else token_count

        span.set_attribute("token_count", token_count)
    finally:
        try:
            otel_context.detach(ctx)
        except ValueError:
            pass
        span.end()
