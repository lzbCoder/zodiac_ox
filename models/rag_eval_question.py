from datetime import datetime
from sqlalchemy import String, Text, Boolean, DateTime, Integer, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import ARRAY
from database import Base


class RagEvalQuestion(Base):
    __tablename__ = "rag_eval_questions"
    __table_args__ = {"schema": "root"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("root.rag_eval_datasets.id", ondelete="CASCADE"), nullable=False)
    kb_id: Mapped[int] = mapped_column(Integer, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    standard_answer: Mapped[str | None] = mapped_column(Text)
    standard_doc_ids: Mapped[list | None] = mapped_column(ARRAY(Integer))
    standard_chunk_ids: Mapped[list | None] = mapped_column(ARRAY(Integer))
    difficulty: Mapped[str] = mapped_column(String(20), default="medium")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)

    dataset = relationship("RagEvalDataset", back_populates="questions")
