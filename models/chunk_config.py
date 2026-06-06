from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class ChunkConfig(Base):
    __tablename__ = "chunk_configs"
    __table_args__ = {"schema": "root"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(ForeignKey("root.knowledge_bases.id", ondelete="CASCADE"), unique=True, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, default=1000)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=100)
    split_separator: Mapped[str] = mapped_column(Text, default="\\n\\n")
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
