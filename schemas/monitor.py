from datetime import datetime
from pydantic import BaseModel, Field


class TraceRequest(BaseModel):
    chat_id: str = Field(..., min_length=1, max_length=64)
    session_id: str | None = None
    kb_id: int | None = None
    query: str | None = None
    answer: str | None = None
    retrieved_chunk_ids: str | None = None
    used_chunk_ids: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    search_cost_ms: int = 0
    llm_cost_ms: int = 0
    total_cost_ms: int = 0
    llm_model: str | None = None
    feedback: str | None = None
    status: str = "success"


class ChunkDetailItem(BaseModel):
    chunk_id: int
    similarity_score: float | None = None
    rank_num: int | None = None
    is_used: int = 0


class TraceSearchRequest(BaseModel):
    chat_id: str = Field(..., min_length=1, max_length=64)
    retrieved_chunk_ids: str | None = None
    used_chunk_ids: str | None = None
    search_cost_ms: int = 0
    chunks: list[ChunkDetailItem] = []


class OverviewResponse(BaseModel):
    total_conversations: int = 0
    success_conversations: int = 0
    avg_search_cost_ms: float = 0.0
    avg_llm_cost_ms: float = 0.0
    avg_total_tokens: float = 0.0
    avg_chunks_count: float = 0.0


class TrendPoint(BaseModel):
    time: str
    value: float


class TrendResponse(BaseModel):
    data: list[TrendPoint] = []


class ChatListItem(BaseModel):
    id: int
    chat_id: str
    query: str
    kb_id: int | None = None
    llm_model: str | None = None
    status: str
    total_cost_ms: int = 0
    total_tokens: int = 0
    feedback: str | None = None
    create_time: datetime | str


class ChatListResponse(BaseModel):
    items: list[ChatListItem] = []
    total: int = 0


class ChatDetailResponse(BaseModel):
    id: int
    chat_id: str
    session_id: str | None = None
    kb_id: int | None = None
    query: str
    answer: str | None = None
    retrieved_chunk_ids: str | None = None
    used_chunk_ids: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    search_cost_ms: int = 0
    llm_cost_ms: int = 0
    total_cost_ms: int = 0
    llm_model: str | None = None
    feedback: str | None = None
    status: str = "success"
    create_time: datetime | str
    chunks: list[ChunkDetailItem] = []


class MonitorFilter(BaseModel):
    start_time: str | None = None
    end_time: str | None = None
    kb_id: str | None = None
    status: str | None = None
    trend_type: str | None = None
    page: int = 1
    page_size: int = 20
    keyword: str | None = None
