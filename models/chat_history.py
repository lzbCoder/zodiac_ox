from datetime import datetime
from sqlalchemy import String, Text, Boolean, Integer, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class ChatHistory(Base):
    __tablename__ = "chat_histories"
    __table_args__ = {"schema": "root"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    kb_id: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), default="qwen3-max")
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    ai_answer: Mapped[str] = mapped_column(Text, nullable=False)
    reference_chunks: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
