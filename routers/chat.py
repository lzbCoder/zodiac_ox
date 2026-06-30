import uuid
import json
import time
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from opentelemetry import trace, context as otel_context
from database import get_db
from schemas.chat import ChatRequest, ReferenceChunk
from services import chat_service, rag_service, document_service, monitor_service
from services.memory_service import get_memory_manager
from models.system_config import SystemConfig
from milvus_client import get_collection
from loguru import logger

router = APIRouter(prefix="/api/chat", tags=["RAG问答"])


# ═══════════════════════════════════════════════════════════════
# 私有辅助函数
# ═══════════════════════════════════════════════════════════════

async def _load_retrieval_config(db: AsyncSession) -> dict:
    """从 system_configs 表加载检索参数"""
    stmt = select(SystemConfig).where(SystemConfig.config_key.in_([
        "retrieval.dense_top_k", "retrieval.sparse_top_k", "retrieval.final_top_k",
    ]))
    result = await db.execute(stmt)
    rows = {r.config_key: int(r.config_value) for r in result.scalars().all()}
    return {
        "dense_top_k": rows.get("retrieval.dense_top_k", 5),
        "sparse_top_k": rows.get("retrieval.sparse_top_k", 5),
        "final_top_k": rows.get("retrieval.final_top_k", 5),
    }


async def _is_memory_enabled(db: AsyncSession) -> bool:
    """检查短期/长期记忆开关是否启用，默认开启"""
    stmt = select(SystemConfig).where(SystemConfig.config_key == "memory.enabled")
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        return True
    return row.config_value.lower() in ("true", "1", "yes")


async def _enrich_hits(db: AsyncSession, hits: list[dict]) -> tuple[list[dict], list[ReferenceChunk]]:
    """根据Milvus检索结果从PG加载chunk完整内容，返回(上下文列表, 引用列表)"""
    if not hits:
        return [], []

    chunks_content = []
    references = []

    # 提取唯一doc_id，避免重复查询
    unique_doc_ids = list(set(h["doc_id"] for h in hits))
    doc_map = {}       # doc_id → Document
    chunk_map = {}     # doc_id → {chunk_id → DocumentChunk}

    for doc_id in unique_doc_ids:
        doc_map[doc_id] = await document_service.get_document(db, doc_id)
        chunks = await document_service.get_chunks_by_doc(db, doc_id)
        chunk_map[doc_id] = {c.id: c for c in chunks}

    for hit in hits:
        doc_id = hit["doc_id"]
        chunk_id = hit["chunk_id"]
        doc = doc_map.get(doc_id)
        c = chunk_map.get(doc_id, {}).get(chunk_id)
        if c:
            chunks_content.append({
                "doc_id": doc_id,
                "filename": doc.filename if doc else "未知文档",
                "content": c.content,
                "page_num": c.page_num,
            })
            references.append(ReferenceChunk(
                doc_id=doc_id,
                filename=doc.filename if doc else "未知",
                chunk_id=c.chunk_index,
                content=c.content[:200],
                page_num=c.page_num,
                score=hit["score"],
            ))

    return chunks_content, references


async def _load_prompt_with_memory(db: AsyncSession, memory_context: str) -> tuple[str, str | None]:
    """加载system/user prompt模板，并将记忆上下文注入system prompt"""
    stmt = select(SystemConfig).where(SystemConfig.config_key.in_(["system_prompt", "user_prompt"]))
    result = await db.execute(stmt)
    rows = {r.config_key: r.config_value for r in result.scalars().all()}
    sys_prompt = rows.get("system_prompt")
    usr_prompt_template = rows.get("user_prompt")

    if memory_context:
        if sys_prompt:
            sys_prompt = sys_prompt + "\n\n" + memory_context
        else:
            sys_prompt = "你是一个知识库问答助手。\n\n" + memory_context

    return sys_prompt, usr_prompt_template


def _build_search_trace_data(hits: list[dict]) -> list[dict]:
    """将检索结果转为监控用的chunk详情列表"""
    return [
        {
            "chunk_id": h["chunk_id"],
            "similarity_score": round(h["score"], 4),
            "rank_num": i,
            "is_used": 1,
        }
        for i, h in enumerate(hits, 1)
    ]


# ═══════════════════════════════════════════════════════════════
# 核心端点
# ═══════════════════════════════════════════════════════════════

@router.post("/ask")
async def ask(data: ChatRequest, db: AsyncSession = Depends(get_db)):
    # ── 1. 初始化OTel追踪 + 请求标识 ──────────────────────────
    rag_span = trace.get_tracer(__name__).start_span("rag_chat")
    rag_span.set_attribute("session_id", data.session_id or "")
    rag_span.set_attribute("kb_id", data.kb_id)
    rag_span.set_attribute("search_mode", data.search_mode)
    rag_span.set_attribute("model", data.model_name)
    rag_ctx = otel_context.attach(trace.set_span_in_context(rag_span))

    chat_id = uuid.uuid4().hex[:16]
    session_id = data.session_id or uuid.uuid4().hex[:16]
    user_id = data.user_id or "admin"
    t_start = time.perf_counter()

    collection = get_collection()
    collection.load()

    # ── 2. 加载记忆上下文（会话短期 + 用户长期向量）───────────
    memory_manager = get_memory_manager()
    memory_enabled = await _is_memory_enabled(db)
    if memory_enabled:
        memory_context = await memory_manager.load_memory_context(
            query=data.query, user_id=user_id, session_id=session_id,
        )
    else:
        memory_context = ""

    # ── 3. 异步推送查询开始trace ──────────────────────────────
    asyncio_create_task_safe(monitor_service.push_trace({
        "chat_id": chat_id, "session_id": session_id,
        "kb_id": data.kb_id, "query": data.query,
        "llm_model": data.model_name, "status": "success",
    }))

    # ── 4. 向量检索 ───────────────────────────────────────────
    ret_config = await _load_retrieval_config(db)
    t_search_start = time.perf_counter()
    hits = await rag_service.search_vectors(
        collection=collection, query=data.query, kb_id=data.kb_id,
        search_mode=data.search_mode,
        dense_top_k=ret_config["dense_top_k"],
        sparse_top_k=ret_config["sparse_top_k"],
        final_top_k=ret_config["final_top_k"],
    )
    search_cost_ms = int((time.perf_counter() - t_search_start) * 1000)

    # ── 5. 异步推送检索trace ──────────────────────────────────
    retrieved_ids = ",".join(str(h["chunk_id"]) for h in hits)
    asyncio_create_task_safe(monitor_service.push_trace({
        "chat_id": chat_id,
        "retrieved_chunk_ids": retrieved_ids,
        "used_chunk_ids": retrieved_ids,
        "search_cost_ms": search_cost_ms,
        "chunks": _build_search_trace_data(hits),
    }))

    # ── 6. 加载chunk内容 + 提示词（含记忆注入）─────────────────
    chunks_content, references = await _enrich_hits(db, hits)
    sys_prompt, usr_prompt_template = await _load_prompt_with_memory(db, memory_context)

    # ── 7. SSE流式生成 ────────────────────────────────────────
    async def stream_answer():
        nonlocal rag_span, rag_ctx
        span_ended = False
        full_answer = ""
        token_count = 0
        llm_start = time.perf_counter()
        try:
            # 7a. LLM流式调用
            try:
                async for token in rag_service.generate_answer_stream(
                    data.query, chunks_content, data.model_name,
                    system_prompt=sys_prompt,
                    user_prompt_template=usr_prompt_template,
                ):
                    full_answer += token
                    token_count += 1
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                status = "success"
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                status = "fail"
                rag_span.set_attribute("error", True)
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

            llm_cost_ms = int((time.perf_counter() - llm_start) * 1000)
            total_cost_ms = int((time.perf_counter() - t_start) * 1000)

            # 7b. 推送生成trace
            asyncio_create_task_safe(monitor_service.push_trace({
                "chat_id": chat_id,
                "answer": full_answer if status == "success" else None,
                "prompt_tokens": 0,
                "completion_tokens": token_count,
                "total_tokens": token_count,
                "llm_cost_ms": llm_cost_ms,
                "total_cost_ms": total_cost_ms,
                "status": status,
            }))

            # 7c. 保存对话历史（chat_histories表）
            refs_data = [r.model_dump() for r in references]
            if data.message_id:
                msg = await chat_service.update_message(
                    db=db, message_id=data.message_id,
                    ai_answer=full_answer, reference_chunks=refs_data,
                )
            else:
                msg = await chat_service.save_message(
                    db=db, session_id=session_id, kb_id=data.kb_id,
                    model_name=data.model_name, user_query=data.query,
                    ai_answer=full_answer, reference_chunks=refs_data,
                )

            # 7d. 保存到双层记忆（fire-and-forget，不阻塞响应）
            if memory_enabled:
                asyncio_create_task_safe(memory_manager.save_interaction(
                    query=data.query, answer=full_answer,
                    user_id=user_id, session_id=session_id,
                ))

            # 7e. 发送SSE收尾事件
            yield f"data: {json.dumps({'type': 'references', 'data': refs_data})}\n\n"
            yield f"data: {json.dumps({'type': 'session_id', 'data': session_id})}\n\n"
            yield f"data: {json.dumps({'type': 'message_id', 'data': msg.id if msg else data.message_id})}\n\n"
            yield f"data: {json.dumps({'type': 'chat_id', 'data': chat_id})}\n\n"
            yield "data: [DONE]\n\n"

            # 7f. 结束OTel span
            rag_span.set_attribute("total_cost_ms", total_cost_ms)
            rag_span.set_attribute("status", status)
            _detach_safe(rag_ctx)
            rag_span.end()
            span_ended = True
        finally:
            if not span_ended:
                _detach_safe(rag_ctx)
                rag_span.end()

    return StreamingResponse(stream_answer(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════
# 记忆管理端点
# ═══════════════════════════════════════════════════════════════

@router.delete("/memory/session/{session_id}")
async def clear_session_memory(session_id: str):
    """清除单会话短期记忆"""
    await get_memory_manager().clear_session_memory(session_id)
    return {"session_id": session_id, "status": "cleared"}


@router.delete("/memory/user/{user_id}")
async def clear_user_memory(user_id: str):
    """清除用户全部记忆（所有会话 + 全局向量记忆）"""
    await get_memory_manager().clear_user_memory(user_id)
    return {"user_id": user_id, "status": "cleared"}


@router.get("/memory/status")
async def memory_status(session_id: str | None = None, user_id: str = "admin"):
    """查询记忆状态"""
    return await get_memory_manager().get_memory_status(
        user_id=user_id, session_id=session_id,
    )


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _detach_safe(token):
    """安全detach OTel context，忽略跨上下文错误"""
    try:
        otel_context.detach(token)
    except ValueError:
        pass


def asyncio_create_task_safe(coro):
    """fire-and-forget 后台协程"""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        pass
