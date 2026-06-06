from datetime import datetime
from sqlalchemy import String, Integer, DateTime, Float, func
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class ChatChunkDetail(Base):
    __tablename__ = "chat_chunk_details"
    __table_args__ = {"schema": "root"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chat_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    chunk_id: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity_score: Mapped[float | None] = mapped_column(Float)
    rank_num: Mapped[int | None] = mapped_column(Integer)
    is_used: Mapped[int] = mapped_column(Integer, default=0)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
