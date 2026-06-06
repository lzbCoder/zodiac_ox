from datetime import datetime
from sqlalchemy import String, DateTime, Integer, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class RagEvalConfig(Base):
    __tablename__ = "rag_eval_configs"
    __table_args__ = {"schema": "root"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(ForeignKey("root.knowledge_bases.id", ondelete="CASCADE"), unique=True, nullable=False)
    default_top_k: Mapped[int] = mapped_column(Integer, default=5)
    default_retriever_mode: Mapped[str] = mapped_column(String(20), default="normal")
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
