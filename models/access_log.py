from datetime import datetime
from sqlalchemy import String, Text, Integer, BigInteger, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class AccessLog(Base):
    """客户端访问记录表 — 按 IP/会话首次去重写入（详见 services/access_log_service.py）。"""

    __tablename__ = "access_log"
    __table_args__ = {"schema": "root"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    client_ip: Mapped[str | None] = mapped_column(String(64), index=True)
    session_id: Mapped[str | None] = mapped_column(String(64))
    user_id: Mapped[str | None] = mapped_column(String(64))
    method: Mapped[str | None] = mapped_column(String(10))
    path: Mapped[str | None] = mapped_column(String(500))
    status_code: Mapped[int | None] = mapped_column(Integer)
    user_agent: Mapped[str | None] = mapped_column(Text)
    referer: Mapped[str | None] = mapped_column(String(500))
    cost_ms: Mapped[int | None] = mapped_column(BigInteger)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
