from datetime import datetime
from pydantic import BaseModel, Field


class ChunkConfigInput(BaseModel):
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=100, ge=0, le=2000)
    split_separator: str = Field(default="\n\n")


class DocumentResponse(BaseModel):
    id: int
    kb_id: int
    kb_name: str = ""
    filename: str
    file_type: str
    file_path: str
    file_size: int
    page_count: int
    upload_status: str
    vector_status: str
    chunk_count: int
    created_at: datetime
    is_deleted: bool

    model_config = {"from_attributes": True}


class ChunkPreview(BaseModel):
    chunk_index: int
    content: str
    page_num: int = 0


class DocumentPreview(BaseModel):
    filename: str
    file_type: str
    content: str
    chunks: list[ChunkPreview] = []
