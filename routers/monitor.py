import io
from urllib.parse import quote
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from schemas.monitor import (
    TraceRequest, TraceSearchRequest,
    OverviewResponse, TrendResponse, ChatListResponse, ChatDetailResponse,
    ChatListItem,
)
from services import monitor_service

router = APIRouter(prefix="/api/rag/monitor", tags=["RAG监控"])


@router.post("/trace")
async def collect_trace(data: TraceRequest):
    """收集 trace 数据（分阶段提交），写入 Redis 缓冲区。"""
    payload = data.model_dump(exclude_none=True)
    ok = await monitor_service.push_trace(payload)
    if not ok:
        raise HTTPException(503, "Trace service unavailable")
    return {"code": 0, "message": "ok"}


@router.post("/trace/search")
async def collect_trace_search(data: TraceSearchRequest):
    """收集检索阶段 trace 数据，含 chunk 详情。"""
    payload = data.model_dump(exclude_none=True)
    ok = await monitor_service.push_trace(payload)
    if not ok:
        raise HTTPException(503, "Trace service unavailable")
    return {"code": 0, "message": "ok"}


def _parse_filters(
    startTime: str | None = None,
    endTime: str | None = None,
    kbId: str | None = None,
    status: str | None = None,
    keyword: str | None = None,
) -> dict:
    return {
        "start_time": startTime,
        "end_time": endTime,
        "kb_id": kbId,
        "status": status,
        "keyword": keyword,
    }


@router.get("/overview")
async def get_overview(
    startTime: str | None = Query(None),
    endTime: str | None = Query(None),
    kbId: str | None = Query(None),
    status: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    filters = _parse_filters(startTime, endTime, kbId, status)
    data = await monitor_service.get_overview(db, filters)
    return {"code": 0, "data": data}


@router.get("/trend")
async def get_trend(
    startTime: str | None = Query(None),
    endTime: str | None = Query(None),
    kbId: str | None = Query(None),
    status: str | None = Query(None),
    trendType: str = Query(..., description="chatCount|searchCost|llmCost|token"),
    db: AsyncSession = Depends(get_db),
):
    filters = _parse_filters(startTime, endTime, kbId, status)
    data = await monitor_service.get_trend(db, filters, trendType)
    return {"code": 0, "data": data}


@router.get("/feedback-distribution")
async def get_feedback_distribution(
    startTime: str | None = Query(None),
    endTime: str | None = Query(None),
    kbId: str | None = Query(None),
    status: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    filters = _parse_filters(startTime, endTime, kbId, status)
    data = await monitor_service.get_feedback_distribution(db, filters)
    return {"code": 0, "data": data}


@router.get("/kb-distribution")
async def get_kb_distribution(
    startTime: str | None = Query(None),
    endTime: str | None = Query(None),
    kbId: str | None = Query(None),
    status: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    filters = _parse_filters(startTime, endTime, kbId, status)
    data = await monitor_service.get_kb_distribution(db, filters)
    return {"code": 0, "data": data}


@router.get("/chat-list")
async def get_chat_list(
    startTime: str | None = Query(None),
    endTime: str | None = Query(None),
    kbId: str | None = Query(None),
    status: str | None = Query(None),
    keyword: str | None = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    filters = _parse_filters(startTime, endTime, kbId, status)
    items, total = await monitor_service.get_chat_list(db, filters, page, size, keyword)
    return {"code": 0, "data": {"items": items, "total": total}}


@router.get("/chat-detail")
async def get_chat_detail(
    chatId: str = Query(..., min_length=1, max_length=64),
    db: AsyncSession = Depends(get_db),
):
    detail = await monitor_service.get_chat_detail(db, chatId)
    if not detail:
        raise HTTPException(404, "Chat not found")
    return {"code": 0, "data": detail}


@router.get("/export")
async def export_chats(
    startTime: str | None = Query(None),
    endTime: str | None = Query(None),
    kbId: str | None = Query(None),
    status: str | None = Query(None),
    keyword: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    from openpyxl import Workbook

    filters = _parse_filters(startTime, endTime, kbId, status, keyword)
    rows = await monitor_service.export_chats(db, filters)

    wb = Workbook()
    ws = wb.active
    ws.title = "对话明细"
    headers = [
        "对话ID", "会话ID", "知识库ID", "Query", "Answer",
        "Prompt Tokens", "Completion Tokens", "Total Tokens",
        "检索耗时(ms)", "LLM耗时(ms)", "总耗时(ms)", "LLM模型",
        "反馈", "状态", "时间",
    ]
    ws.append(headers)

    for r in rows:
        ws.append([
            r["chat_id"], r["session_id"], r["kb_id"],
            r["query"], r["answer"],
            r["prompt_tokens"], r["completion_tokens"], r["total_tokens"],
            r["search_cost_ms"], r["llm_cost_ms"], r["total_cost_ms"],
            r["llm_model"],
            r["feedback"], r["status"], r["create_time"],
        ])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"RAG监控_对话明细_{now}.xlsx"
    encoded_filename = quote(filename)

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"},
    )
