import uuid
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from models.chat_history import ChatHistory


async def save_message(
    db: AsyncSession,
    session_id: str,
    kb_id: int,
    model_name: str,
    user_query: str,
    ai_answer: str,
    reference_chunks: list | None = None,
) -> ChatHistory:
    msg = ChatHistory(
        session_id=session_id,
        kb_id=kb_id,
        model_name=model_name,
        user_query=user_query,
        ai_answer=ai_answer,
        reference_chunks=reference_chunks,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def get_sessions(db: AsyncSession, page: int = 1, page_size: int = 20) -> tuple[list[dict], int]:
    # 获取唯一会话及其首条消息
    subq = (
        select(
            ChatHistory.session_id,
            func.min(ChatHistory.id).label("min_id"),
            func.count(ChatHistory.id).label("msg_count"),
            func.max(ChatHistory.created_at).label("last_time"),
        )
        .where(ChatHistory.is_deleted == False)
        .group_by(ChatHistory.session_id)
        .subquery()
    )

    stmt = (
        select(ChatHistory, subq.c.msg_count, subq.c.last_time)
        .join(subq, (ChatHistory.id == subq.c.min_id))
        .order_by(subq.c.last_time.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    count_stmt = select(func.count()).select_from(subq)

    total = (await db.execute(count_stmt)).scalar() or 0
    result = await db.execute(stmt)
    rows = result.all()

    sessions = []
    for msg, msg_count, last_time in rows:
        sessions.append({
            "session_id": msg.session_id,
            "kb_id": msg.kb_id,
            "first_query": msg.user_query[:100],
            "message_count": msg_count,
            "created_at": last_time,
        })
    return sessions, total


async def get_session_messages(db: AsyncSession, session_id: str) -> list[ChatHistory]:
    stmt = (
        select(ChatHistory)
        .where(ChatHistory.session_id == session_id, ChatHistory.is_deleted == False)
        .order_by(ChatHistory.created_at)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def update_message(
    db: AsyncSession,
    message_id: int,
    ai_answer: str,
    reference_chunks: list | None = None,
) -> ChatHistory | None:
    msg = await db.get(ChatHistory, message_id)
    if not msg or msg.is_deleted:
        return None
    msg.ai_answer = ai_answer
    msg.reference_chunks = reference_chunks
    await db.commit()
    await db.refresh(msg)
    return msg


async def delete_message(db: AsyncSession, msg_id: int) -> bool:
    stmt = select(ChatHistory).where(ChatHistory.id == msg_id, ChatHistory.is_deleted == False)
    result = await db.execute(stmt)
    msg = result.scalar_one_or_none()
    if not msg:
        return False
    msg.is_deleted = True
    await db.commit()
    return True


async def clear_session(db: AsyncSession, session_id: str):
    stmt = (
        update(ChatHistory)
        .where(ChatHistory.session_id == session_id)
        .values(is_deleted=True)
    )
    await db.execute(stmt)
    await db.commit()


async def export_session_markdown(db: AsyncSession, session_id: str) -> str:
    messages = await get_session_messages(db, session_id)
    lines = ["# 对话导出\n"]
    for msg in messages:
        lines.append(f"## 用户问题 ({msg.created_at.isoformat()})")
        lines.append(msg.user_query)
        lines.append("\n## AI 回答")
        lines.append(msg.ai_answer)
        lines.append("\n---\n")
    return "\n".join(lines)
