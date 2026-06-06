from datetime import datetime
from sqlalchemy import String, Text, Float, Boolean, DateTime, Integer, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import ARRAY
from database import Base


class RagEvalResult(Base):
    __tablename__ = "rag_eval_results"
    __table_args__ = {"schema": "root"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("root.rag_eval_tasks.id", ondelete="CASCADE"), nullable=False)
    qid: Mapped[int] = mapped_column(Integer, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[list | None] = mapped_column(ARRAY(Integer))
    retrieved_doc_ids: Mapped[list | None] = mapped_column(ARRAY(Integer))
    recall: Mapped[float | None] = mapped_column(Float)
    precision: Mapped[float | None] = mapped_column(Float)
    hit: Mapped[bool | None] = mapped_column(Boolean)
    rank: Mapped[int | None] = mapped_column(Integer)
    mrr: Mapped[float | None] = mapped_column(Float)
    retrieve_time: Mapped[float | None] = mapped_column(Float)
    answer_time: Mapped[float | None] = mapped_column(Float)
    answer: Mapped[str | None] = mapped_column(Text)
    context_precision: Mapped[float | None] = mapped_column(Float)
    context_recall: Mapped[float | None] = mapped_column(Float)
    faithfulness: Mapped[float | None] = mapped_column(Float)
    answer_relevancy: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
