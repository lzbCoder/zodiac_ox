from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class ChatTrace(Base):
    __tablename__ = "chat_traces"
    __table_args__ = {"schema": "root"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chat_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64))
    kb_id: Mapped[int] = mapped_column(Integer, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    retrieved_chunk_ids: Mapped[str | None] = mapped_column(Text)
    used_chunk_ids: Mapped[str | None] = mapped_column(Text)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    search_cost_ms: Mapped[int] = mapped_column(Integer, default=0)
    llm_cost_ms: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_ms: Mapped[int] = mapped_column(Integer, default=0)
    llm_model: Mapped[str | None] = mapped_column(String(64))
    feedback: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="success")
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True, comment="系统提示词")
    user_prompt: Mapped[str | None] = mapped_column(Text, nullable=True, comment="用户提示词")
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
