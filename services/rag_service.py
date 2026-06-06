from typing import AsyncGenerator
import asyncio
from pymilvus import Collection, AnnSearchRequest, RRFRanker
from dashscope import TextEmbedding
from openai import AsyncOpenAI
from opentelemetry import trace, context as otel_context
from config import DASHSCOPE_API_KEY, EMBEDDING_MODEL
from services.sparse_embedding import JiebaSparseEmbedding

_sparse_encoder = JiebaSparseEmbedding()

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
            model=EMBEDDING_MODEL,
            input=query[:2048],
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
            hits.append({
                "id": hit.id,
                "doc_id": hit.entity.get("doc_id"),
                "chunk_id": hit.entity.get("chunk_id"),
                "kb_id": hit.entity.get("kb_id"),
                "score": hit.score,
            })
        span.set_attribute("retrieved_count", len(hits))
        return hits[:final_top_k]


async def generate_answer_stream(
    query: str,
    context_chunks: list[dict],
    model: str = "qwen3-max",
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> AsyncGenerator[str, None]:
    span = trace.get_tracer(__name__).start_span("llm_generate")
    span.set_attribute("model", model)
    span.set_attribute("context_chunks", len(context_chunks))
    span.set_attribute("query_length", len(query))
    ctx = otel_context.attach(trace.set_span_in_context(span))

    try:
        context_text = "\n\n---\n\n".join(
            f"[来源文档ID:{c['doc_id']}] {c['content']}" for c in context_chunks
        )

        if system_prompt is None:
            system_prompt = (
                "你是一个知识库问答助手。请根据提供的文档片段回答用户问题。"
                "如果文档片段不足以回答问题，请如实说明。回答时请引用具体的来源。"
            )

        if user_prompt_template is None:
            user_prompt = f"文档片段：\n{context_text}\n\n用户问题：{query}\n\n请根据以上文档片段回答问题："
        else:
            user_prompt = user_prompt_template.replace("{context}", context_text).replace("{query}", query)

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
        )

        token_count = 0
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token_count += 1
                yield chunk.choices[0].delta.content

        span.set_attribute("token_count", token_count)
    finally:
        try:
            otel_context.detach(ctx)
        except ValueError:
            pass
        span.end()
