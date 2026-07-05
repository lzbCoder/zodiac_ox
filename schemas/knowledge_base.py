from datetime import datetime
from pydantic import BaseModel, Field


class KBCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None


class KBUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None


class KBResponse(BaseModel):
    id: int
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    doc_count: int = 0
    vector_status: str = "normal"

    model_config = {"from_attributes": True}
