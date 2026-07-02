from datetime import datetime
from pydantic import BaseModel, Field


# ── Dataset ──────────────────────────────────────────────

class RagEvalDatasetCreate(BaseModel):
    kb_id: int
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    created_by: str | None = None


class RagEvalDatasetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None


class RagEvalDatasetResponse(BaseModel):
    id: int
    kb_id: int
    kb_name: str = ""
    name: str
    description: str | None = None
    total_questions: int = 0
    created_at: datetime
    updated_at: datetime | None = None
    created_by: str | None = None
    is_deleted: bool = False

    model_config = {"from_attributes": True}


# ── Question ─────────────────────────────────────────────

class RagEvalQuestionItem(BaseModel):
    """Excel/CSV 导入的单行数据。"""
    query: str
    standard_answer: str | None = None
    standard_doc_ids: list[int] | None = None
    standard_chunk_ids: list[int] | None = None
    difficulty: str = "medium"


class RagEvalQuestionImport(BaseModel):
    kb_id: int
    questions: list[RagEvalQuestionItem]


class RagEvalQuestionResponse(BaseModel):
    id: int
    dataset_id: int
    kb_id: int
    query: str
    standard_answer: str | None = None
    standard_doc_ids: list[int] | None = None
    standard_chunk_ids: list[int] | None = None
    difficulty: str = "medium"
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Task ─────────────────────────────────────────────────

class RagEvalTaskCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    task_type: str = Field(default="manual")  # manual | chat_sample
    dataset_id: int | None = None
    kb_id: int
    top_k: int = Field(default=5, ge=1, le=100)
    retriever_mode: str = Field(default="normal")
    model_name: str | None = None
    enable_ragas: bool = False
    eval_model: str | None = None
    # 聊天抽样配置
    sample_time_start: datetime | None = None
    sample_time_end: datetime | None = None
    sample_count: int = Field(default=10, ge=1, le=100)
    sample_strategy: str = Field(default="random")  # random | latest


class RagEvalTaskResponse(BaseModel):
    id: int
    name: str
    task_type: str = "manual"
    dataset_id: int | None = None
    kb_id: int
    dataset_name: str = ""
    kb_name: str = ""
    top_k: int = 5
    retriever_mode: str = "normal"
    model_name: str | None = None
    eval_model: str | None = None
    sample_count: int | None = None
    sample_strategy: str | None = None
    status: str = "pending"
    progress: int = 0
    enable_ragas: bool = False
    recall: float | None = None
    precision: float | None = None
    hit_rate: float | None = None
    mrr: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    cost_seconds: float | None = None
    created_at: datetime
    finished_at: datetime | None = None

    model_config = {"from_attributes": True}


# ── Result ───────────────────────────────────────────────

class RagEvalResultResponse(BaseModel):
    id: int
    task_id: int
    qid: int
    query: str
    retrieved_chunk_ids: list[int] | None = None
    retrieved_doc_ids: list[int] | None = None
    recall: float | None = None
    precision: float | None = None
    hit: bool | None = None
    rank: int | None = None
    mrr: float | None = None
    retrieve_time: float | None = None
    answer_time: float | None = None
    answer: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None

    model_config = {"from_attributes": True}


# ── Report ───────────────────────────────────────────────

class RagEvalReportData(BaseModel):
    task: RagEvalTaskResponse
    total_questions: int
    results: list[RagEvalResultResponse]


# ── Config ───────────────────────────────────────────────

class RagEvalConfigUpdate(BaseModel):
    default_top_k: int | None = Field(default=None, ge=1, le=100)
    default_retriever_mode: str | None = None


class RagEvalConfigResponse(BaseModel):
    id: int
    kb_id: int
    default_top_k: int = 5
    default_retriever_mode: str = "normal"
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


# ── Label Task ────────────────────────────────────────────

class LabelQueryItem(BaseModel):
    query: str
    standard_answer: str | None = None


class RagEvalLabelTaskCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    kb_id: int
    top_k: int = Field(default=5, ge=1, le=100)
    description: str | None = None
    created_by: str | None = None
    queries: list[LabelQueryItem]


class RagEvalLabelTaskUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = None


class RagEvalLabelTaskResponse(BaseModel):
    id: int
    name: str
    kb_id: int
    top_k: int = 5
    description: str | None = None
    created_by: str | None = None
    status: str = "in_progress"
    progress: int = 0
    total_details: int = 0
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


# ── Label Detail ──────────────────────────────────────────

class RagEvalLabelDetailResponse(BaseModel):
    id: int
    task_id: int
    query: str
    standard_answer: str | None = None
    standard_chunk_ids: list[int] | None = None
    standard_doc_ids: list[int] | None = None
    status: str = "unannotated"
    annotated_by: str | None = None
    annotated_at: datetime | None = None

    model_config = {"from_attributes": True}


class RagEvalLabelDetailSave(BaseModel):
    standard_chunk_ids: list[int]
    annotated_by: str | None = None


class RagEvalLabelBatchSave(BaseModel):
    annotations: list[dict]
    annotated_by: str | None = None


# ── Candidate Chunk ───────────────────────────────────────

class CandidateChunkOut(BaseModel):
    chunk_id: int
    doc_id: int
    kb_id: int
    content: str
    doc_name: str
    score: float
    page_num: int | None = None


class ChunkCandidatesResponse(BaseModel):
    detail_id: int
    query: str
    standard_answer: str | None = None
    chunks: list[CandidateChunkOut] = []


# ── Label Export ──────────────────────────────────────────

class LabelExportResponse(BaseModel):
    task: RagEvalLabelTaskResponse
    details: list[RagEvalLabelDetailResponse]
