from datetime import datetime
from sqlalchemy import String, Float, Boolean, DateTime, Integer, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class RagEvalTask(Base):
    __tablename__ = "rag_eval_tasks"
    __table_args__ = {"schema": "root"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    task_type: Mapped[str] = mapped_column(String(20), default="manual")
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("root.rag_eval_datasets.id", ondelete="CASCADE"), nullable=True)
    kb_id: Mapped[int] = mapped_column(ForeignKey("root.knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, default=5)
    retriever_mode: Mapped[str] = mapped_column(String(20), default="normal")
    model_name: Mapped[str | None] = mapped_column(String(50))
    eval_model: Mapped[str | None] = mapped_column(String(50))
    sample_time_start: Mapped[datetime | None] = mapped_column(DateTime)
    sample_time_end: Mapped[datetime | None] = mapped_column(DateTime)
    sample_count: Mapped[int] = mapped_column(Integer, default=10)
    sample_strategy: Mapped[str] = mapped_column(String(20), default="random")
    status: Mapped[str] = mapped_column(String(20), default="pending")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    enable_ragas: Mapped[bool] = mapped_column(Boolean, default=False)
    recall: Mapped[float | None] = mapped_column(Float)
    precision: Mapped[float | None] = mapped_column(Float)
    hit_rate: Mapped[float | None] = mapped_column(Float)
    mrr: Mapped[float | None] = mapped_column(Float)
    context_precision: Mapped[float | None] = mapped_column(Float)
    context_recall: Mapped[float | None] = mapped_column(Float)
    faithfulness: Mapped[float | None] = mapped_column(Float)
    answer_relevancy: Mapped[float | None] = mapped_column(Float)
    cost_seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
