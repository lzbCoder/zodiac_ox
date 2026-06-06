from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from schemas.chat import ChatHistoryItem, SessionInfo
from services import chat_service

router = APIRouter(prefix="/api/history", tags=["对话历史"])


@router.get("/sessions")
async def list_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    sessions, total = await chat_service.get_sessions(db, page, page_size)
    return {"items": sessions, "total": total, "page": page, "page_size": page_size}


@router.get("/sessions/{session_id}")
async def get_session_messages(session_id: str, db: AsyncSession = Depends(get_db)):
    messages = await chat_service.get_session_messages(db, session_id)
    return [ChatHistoryItem.model_validate(m) for m in messages]


@router.delete("/messages/{msg_id}")
async def delete_message(msg_id: int, db: AsyncSession = Depends(get_db)):
    ok = await chat_service.delete_message(db, msg_id)
    if not ok:
        raise HTTPException(status_code=404, detail="消息不存在")
    return {"message": "已删除"}


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str, db: AsyncSession = Depends(get_db)):
    await chat_service.clear_session(db, session_id)
    return {"message": "已清空对话"}


@router.get("/sessions/{session_id}/export")
async def export_session(session_id: str, db: AsyncSession = Depends(get_db)):
    md = await chat_service.export_session_markdown(db, session_id)
    return PlainTextResponse(md, media_type="text/markdown", headers={"Content-Disposition": f"attachment; filename=chat_{session_id}.md"})
