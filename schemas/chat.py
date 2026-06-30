from datetime import datetime
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    kb_id: int
    query: str = Field(..., min_length=1)
    session_id: str | None = None
    user_id: str = Field(default="admin")  # 用户标识，用于长期记忆隔离
    model_name: str = Field(default="qwen3-max")
    search_mode: str = Field(default="normal", pattern="^(normal|hybrid)$")
    message_id: int | None = None  # 若提供则更新已有消息，否则插入新消息


class ReferenceChunk(BaseModel):
    doc_id: int
    filename: str
    chunk_id: int
    content: str
    page_num: int = 0
    score: float = 0.0


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    references: list[ReferenceChunk] = []
    model_name: str


class ChatHistoryItem(BaseModel):
    id: int
    session_id: str
    kb_id: int
    model_name: str
    user_query: str
    ai_answer: str
    reference_chunks: list | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionInfo(BaseModel):
    session_id: str
    kb_id: int
    first_query: str
    message_count: int
    created_at: datetime
