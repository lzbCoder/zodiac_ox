import json
import hashlib
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_, Integer, Float, text
from sqlalchemy.ext.asyncio import AsyncSession
from redis_client import get_redis
from models.chat_trace import ChatTrace
from models.chat_chunk_detail import ChatChunkDetail
from loguru import logger

REDIS_TRACE_LIST = "rag_monitor_trace_list"
REDIS_DEDUP_PREFIX = "rag_chat_id:"
REDIS_CACHE_PREFIX = "monitor_cache:"
DEDUP_TTL = 86400  # 24 hours
DEFAULT_CACHE_TTL = 120  # 2 minutes


def _filter_hash(filters: dict) -> str:
    raw = json.dumps(filters, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


async def push_trace(data: dict) -> bool:
    """推送 trace 数据到 Redis 缓冲区。成功返回 True，Redis 失败返回 False。"""
    try:
        r = await get_redis()
        await r.rpush(REDIS_TRACE_LIST, json.dumps(data, default=str))
        return True
    except Exception as e:
        logger.warning(f"Redis push_trace failed: {e}")
        return False


async def check_and_mark_dedup(chat_id: str) -> bool:
    """检查 chat_id 是否已被追踪。如果是重复的（应跳过）返回 True。"""
    try:
        r = await get_redis()
        key = f"{REDIS_DEDUP_PREFIX}{chat_id}"
        # SET NX：key 不存在时返回 True（新创建），已存在返回 None
        was_set = await r.set(key, "1", nx=True, ex=DEDUP_TTL)
        return not was_set  # True = 重复，False = 新记录
    except Exception as e:
        logger.warning(f"Redis dedup check failed: {e}")
        return False  # 失败时不阻塞 — 放行


async def cache_get(key: str) -> dict | None:
    try:
        r = await get_redis()
        data = await r.get(f"{REDIS_CACHE_PREFIX}{key}")
        return json.loads(data) if data else None
    except Exception:
        return None


async def cache_set(key: str, data: dict, ttl: int = DEFAULT_CACHE_TTL):
    try:
        r = await get_redis()
        await r.setex(f"{REDIS_CACHE_PREFIX}{key}", ttl, json.dumps(data, default=str))
    except Exception as e:
        logger.warning(f"Redis cache_set failed: {e}")


async def flush_traces(db: AsyncSession):
    """批量将 trace 数据从 Redis 刷入 PostgreSQL。"""
    try:
        r = await get_redis()
        # 原子性地读取所有项并清空列表
        items = []
        while True:
            item = await r.lpop(REDIS_TRACE_LIST)
            if item is None:
                break
            items.append(item)
            if len(items) >= 500:  # 每次刷新周期的批量上限
                break
        if not items:
            return

        for raw in items:
            try:
                data = json.loads(raw)
                chat_id = data.get("chat_id")
                if not chat_id:
                    continue

                # Upsert chat_trace — 合并部分数据
                existing = (await db.execute(
                    select(ChatTrace).where(ChatTrace.chat_id == chat_id)
                )).scalar_one_or_none()

                if existing:
                    # 合并字段：仅更新非空值
                    for field in [
                        "session_id", "kb_id", "query", "answer",
                        "retrieved_chunk_ids", "used_chunk_ids",
                        "prompt_tokens", "completion_tokens", "total_tokens",
                        "search_cost_ms", "llm_cost_ms", "total_cost_ms",
                        "llm_model", "feedback", "status",
                        "system_prompt", "user_prompt",
                    ]:
                        val = data.get(field)
                        if val is not None:
                            setattr(existing, field, val)
                else:
                    trace = ChatTrace(
                        chat_id=chat_id,
                        session_id=data.get("session_id"),
                        kb_id=data.get("kb_id", 0),
                        query=data.get("query", ""),
                        answer=data.get("answer"),
                        retrieved_chunk_ids=data.get("retrieved_chunk_ids"),
                        used_chunk_ids=data.get("used_chunk_ids"),
                        prompt_tokens=data.get("prompt_tokens", 0),
                        completion_tokens=data.get("completion_tokens", 0),
                        total_tokens=data.get("total_tokens", 0),
                        search_cost_ms=data.get("search_cost_ms", 0),
                        llm_cost_ms=data.get("llm_cost_ms", 0),
                        total_cost_ms=data.get("total_cost_ms", 0),
                        llm_model=data.get("llm_model"),
                        feedback=data.get("feedback"),
                        status=data.get("status", "success"),
                        system_prompt=data.get("system_prompt"),
                        user_prompt=data.get("user_prompt"),
                    )
                    db.add(trace)

                # 处理 chunk 详情（如有）
                chunks = data.get("chunks")
                if chunks and isinstance(chunks, list):
                    # 删除该 chat_id 的旧 chunk 详情
                    await db.execute(
                        ChatChunkDetail.__table__.delete().where(
                            ChatChunkDetail.chat_id == chat_id
                        )
                    )
                    for c in chunks:
                        detail = ChatChunkDetail(
                            chat_id=chat_id,
                            chunk_id=c.get("chunk_id", 0),
                            similarity_score=c.get("similarity_score"),
                            rank_num=c.get("rank_num"),
                            is_used=c.get("is_used", 0),
                        )
                        db.add(detail)

            except Exception as e:
                logger.error(f"Flush trace item failed: {e}, raw={raw[:200]}")

        await db.commit()
        if items:
            logger.info(f"Flushed {len(items)} traces to PostgreSQL")

    except Exception as e:
        logger.error(f"flush_traces failed: {e}")


def _build_filters(query, filters: dict, model=ChatTrace):
    """对查询应用通用过滤条件。"""
    conditions = []
    if filters.get("start_time"):
        try:
            t = datetime.fromisoformat(filters["start_time"])
            conditions.append(model.create_time >= t)
        except ValueError:
            pass
    if filters.get("end_time"):
        try:
            t = datetime.fromisoformat(filters["end_time"])
            conditions.append(model.create_time <= t)
        except ValueError:
            pass
    if filters.get("kb_id"):
        kb_ids = [int(k) for k in filters["kb_id"].split(",") if k.strip().isdigit()]
        if kb_ids:
            conditions.append(model.kb_id.in_(kb_ids))
    if filters.get("status") and filters["status"] != "all":
        conditions.append(model.status == filters["status"])
    if conditions:
        query = query.where(and_(*conditions))
    return query


async def get_overview(db: AsyncSession, filters: dict) -> dict:
    cache_key = f"overview_{_filter_hash(filters)}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    base = select(ChatTrace)
    base = _build_filters(base, filters)

    total_stmt = select(func.count()).select_from(base.subquery())
    success_stmt = select(func.count()).select_from(
        _build_filters(select(ChatTrace), filters).where(ChatTrace.status == "success").subquery()
    )

    stats_stmt = select(
        func.avg(ChatTrace.search_cost_ms),
        func.avg(ChatTrace.llm_cost_ms),
        func.avg(ChatTrace.total_tokens),
        func.avg(
            func.coalesce(
                func.array_length(
                    func.string_to_array(ChatTrace.retrieved_chunk_ids, ","), 1
                ), 0
            )
        ),
    )
    stats_stmt = _build_filters(stats_stmt, filters).where(ChatTrace.status == "success")

    total = (await db.execute(total_stmt)).scalar() or 0
    success = (await db.execute(success_stmt)).scalar() or 0
    stats = (await db.execute(stats_stmt)).one_or_none()

    result = {
        "total_conversations": total,
        "success_conversations": success,
        "avg_search_cost_ms": round(stats[0] or 0, 1),
        "avg_llm_cost_ms": round(stats[1] or 0, 1),
        "avg_total_tokens": round(stats[2] or 0, 1),
        "avg_chunks_count": round(stats[3] or 0, 1),
    }
    await cache_set(cache_key, result)
    return result


async def get_trend(db: AsyncSession, filters: dict, trend_type: str) -> list[dict]:
    cache_key = f"trend_{trend_type}_{_filter_hash(filters)}"
    cached = await cache_get(cache_key)
    if cached:
        return cached.get("data", [])

    # 根据时间范围选择时间桶：小时或天
    start = filters.get("start_time")
    end = filters.get("end_time")
    if start and end:
        try:
            s = datetime.fromisoformat(start)
            e = datetime.fromisoformat(end)
            bucket = "hour" if (e - s) <= timedelta(days=2) else "day"
        except ValueError:
            bucket = "day"
    else:
        bucket = "hour"

    if bucket == "hour":
        time_trunc = func.date_trunc("hour", ChatTrace.create_time)
        fmt = "YYYY-MM-DD HH24:00"
    else:
        time_trunc = func.date_trunc("day", ChatTrace.create_time)
        fmt = "YYYY-MM-DD"

    time_label = func.to_char(time_trunc, fmt)

    if trend_type == "chatCount":
        stmt = (
            select(
                time_label.label("time"),
                func.count().label("total"),
                func.sum(
                    func.cast(ChatTrace.status == "success", Integer)
                ).label("success"),
                func.sum(
                    func.cast(ChatTrace.status == "fail", Integer)
                ).label("fail"),
            )
            .group_by(time_trunc, time_label)
            .order_by(time_trunc)
        )
    elif trend_type == "searchCost":
        stmt = (
            select(
                time_label.label("time"),
                func.avg(ChatTrace.search_cost_ms).label("value"),
            )
            .where(ChatTrace.status == "success")
            .group_by(time_trunc, time_label)
            .order_by(time_trunc)
        )
    elif trend_type == "llmCost":
        stmt = (
            select(
                time_label.label("time"),
                func.avg(ChatTrace.llm_cost_ms).label("value"),
            )
            .where(ChatTrace.status == "success")
            .group_by(time_trunc, time_label)
            .order_by(time_trunc)
        )
    elif trend_type == "token":
        stmt = (
            select(
                time_label.label("time"),
                func.avg(ChatTrace.total_tokens).label("value"),
            )
            .where(ChatTrace.status == "success")
            .group_by(time_trunc, time_label)
            .order_by(time_trunc)
        )
    else:
        return []

    stmt = _build_filters(stmt, filters)
    rows = (await db.execute(stmt)).all()

    if trend_type == "chatCount":
        data = [
            {
                "time": r.time,
                "total": r.total or 0,
                "success": r.success or 0,
                "fail": r.fail or 0,
            }
            for r in rows
        ]
    else:
        data = [{"time": r.time, "value": round(r.value or 0, 1)} for r in rows]

    result = {"data": data}
    await cache_set(cache_key, result)
    return data


async def get_feedback_distribution(db: AsyncSession, filters: dict) -> list[dict]:
    cache_key = f"feedback_{_filter_hash(filters)}"
    cached = await cache_get(cache_key)
    if cached:
        return cached.get("data", [])

    stmt = select(
        func.coalesce(ChatTrace.feedback, "none").label("type"),
        func.count(),
    ).group_by(text("type"))
    stmt = _build_filters(stmt, filters)

    rows = (await db.execute(stmt)).all()
    data = [{"name": r.type, "value": r.count} for r in rows]
    result = {"data": data}
    await cache_set(cache_key, result)
    return data


async def get_kb_distribution(db: AsyncSession, filters: dict) -> list[dict]:
    from models.knowledge_base import KnowledgeBase

    cache_key = f"kb_dist_{_filter_hash(filters)}"
    cached = await cache_get(cache_key)
    if cached:
        return cached.get("data", [])

    stmt = (
        select(ChatTrace.kb_id, KnowledgeBase.name, func.count())
        .join(KnowledgeBase, ChatTrace.kb_id == KnowledgeBase.id, isouter=True)
        .group_by(ChatTrace.kb_id, KnowledgeBase.name)
        .order_by(func.count().desc())
    )
    stmt = _build_filters(stmt, filters)

    rows = (await db.execute(stmt)).all()
    data = [{"kb_id": r.kb_id, "kb_name": r.name or f"KB-{r.kb_id}", "count": r.count} for r in rows]
    result = {"data": data}
    await cache_set(cache_key, result)
    return data


async def get_chat_list(
    db: AsyncSession, filters: dict, page: int = 1, page_size: int = 20, keyword: str | None = None
) -> tuple[list[dict], int]:
    from models.knowledge_base import KnowledgeBase

    base = select(
        ChatTrace.id,
        ChatTrace.chat_id,
        ChatTrace.query,
        ChatTrace.kb_id,
        KnowledgeBase.name.label("kb_name"),
        ChatTrace.status,
        ChatTrace.llm_model,
        ChatTrace.total_cost_ms,
        ChatTrace.total_tokens,
        ChatTrace.feedback,
        ChatTrace.create_time,
    ).join(KnowledgeBase, ChatTrace.kb_id == KnowledgeBase.id, isouter=True)

    base = _build_filters(base, filters)
    if keyword:
        base = base.where(
            and_(
                ChatTrace.query.ilike(f"%{keyword}%")
                | (ChatTrace.chat_id.ilike(f"%{keyword}%"))
            )
        )

    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = base.order_by(ChatTrace.create_time.desc()).offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).all()

    items = []
    for r in rows:
        items.append({
            "id": r.id,
            "chat_id": r.chat_id,
            "query": r.query,
            "kb_id": r.kb_id,
            "kb_name": r.kb_name,
            "llm_model": r.llm_model,
            "status": r.status,
            "total_cost_ms": r.total_cost_ms or 0,
            "total_tokens": r.total_tokens or 0,
            "feedback": r.feedback,
            "create_time": r.create_time.isoformat() if r.create_time else "",
        })
    return items, total


async def get_chat_detail(db: AsyncSession, chat_id: str) -> dict | None:
    trace = (await db.execute(
        select(ChatTrace).where(ChatTrace.chat_id == chat_id)
    )).scalar_one_or_none()
    if not trace:
        return None

    chunks = (await db.execute(
        select(ChatChunkDetail).where(ChatChunkDetail.chat_id == chat_id).order_by(ChatChunkDetail.rank_num)
    )).scalars().all()

    return {
        "id": trace.id,
        "chat_id": trace.chat_id,
        "session_id": trace.session_id,
        "kb_id": trace.kb_id,
        "query": trace.query,
        "answer": trace.answer,
        "retrieved_chunk_ids": trace.retrieved_chunk_ids,
        "used_chunk_ids": trace.used_chunk_ids,
        "prompt_tokens": trace.prompt_tokens or 0,
        "completion_tokens": trace.completion_tokens or 0,
        "total_tokens": trace.total_tokens or 0,
        "search_cost_ms": trace.search_cost_ms or 0,
        "llm_cost_ms": trace.llm_cost_ms or 0,
        "total_cost_ms": trace.total_cost_ms or 0,
        "llm_model": trace.llm_model,
        "feedback": trace.feedback,
        "status": trace.status,
        "system_prompt": trace.system_prompt,
        "user_prompt": trace.user_prompt,
        "create_time": trace.create_time.isoformat() if trace.create_time else "",
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "similarity_score": c.similarity_score,
                "rank_num": c.rank_num,
                "is_used": c.is_used,
            }
            for c in chunks
        ],
    }


async def export_chats(db: AsyncSession, filters: dict) -> list[dict]:
    """导出所有匹配的 trace（无分页）。"""
    stmt = select(ChatTrace).order_by(ChatTrace.create_time.desc())
    stmt = _build_filters(stmt, filters)

    if filters.get("keyword"):
        kw = filters["keyword"]
        stmt = stmt.where(ChatTrace.query.ilike(f"%{kw}%"))

    rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "chat_id": r.chat_id,
            "session_id": r.session_id,
            "kb_id": r.kb_id,
            "query": r.query,
            "answer": r.answer,
            "prompt_tokens": r.prompt_tokens or 0,
            "completion_tokens": r.completion_tokens or 0,
            "total_tokens": r.total_tokens or 0,
            "search_cost_ms": r.search_cost_ms or 0,
            "llm_cost_ms": r.llm_cost_ms or 0,
            "total_cost_ms": r.total_cost_ms or 0,
            "llm_model": r.llm_model,
            "feedback": r.feedback,
            "status": r.status,
            "create_time": r.create_time.isoformat() if r.create_time else "",
        }
        for r in rows
    ]
