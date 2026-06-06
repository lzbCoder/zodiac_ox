from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import ARRAY
from database import Base


class RagEvalLabelDetail(Base):
    __tablename__ = "rag_eval_label_details"
    __table_args__ = {"schema": "root"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("root.rag_eval_label_tasks.id", ondelete="CASCADE"), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    standard_answer: Mapped[str | None] = mapped_column(Text)
    standard_chunk_ids: Mapped[list | None] = mapped_column(ARRAY(Integer))
    standard_doc_ids: Mapped[list | None] = mapped_column(ARRAY(Integer))
    status: Mapped[str] = mapped_column(String(20), default="unannotated")
    annotated_by: Mapped[str | None] = mapped_column(String(100))
    annotated_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
