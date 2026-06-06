from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = {"schema": "root"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(nullable=False)
    doc_id: Mapped[int] = mapped_column(ForeignKey("root.documents.id", ondelete="CASCADE"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_num: Mapped[int] = mapped_column(Integer, default=0)
    start_pos: Mapped[int] = mapped_column(Integer, default=0)
    end_pos: Mapped[int] = mapped_column(Integer, default=0)
    milvus_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    document = relationship("Document", back_populates="chunks")
