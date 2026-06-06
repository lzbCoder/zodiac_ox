from datetime import datetime
from sqlalchemy import String, Text, Boolean, DateTime, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class RagEvalDataset(Base):
    __tablename__ = "rag_eval_datasets"
    __table_args__ = {"schema": "root"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    total_questions: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    created_by: Mapped[str | None] = mapped_column(String(100))
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)

    questions = relationship("RagEvalQuestion", back_populates="dataset", lazy="selectin", cascade="all, delete-orphan")
