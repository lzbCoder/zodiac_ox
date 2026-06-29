from datetime import datetime
from pydantic import BaseModel


class SystemConfigCreate(BaseModel):
    config_key: str
    config_value: str
    description: str | None = None


class SystemConfigUpdate(BaseModel):
    config_value: str
    description: str | None = None


class SystemConfigResponse(BaseModel):
    id: int
    config_key: str
    config_value: str | None
    description: str | None
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConnectionStatus(BaseModel):
    postgresql: bool
    milvus: bool
    redis: bool


class PromptConfigRequest(BaseModel):
    system_prompt: str
    user_prompt: str


class PromptConfigResponse(BaseModel):
    system_prompt: str
    user_prompt: str


class RetrievalConfigRequest(BaseModel):
    dense_top_k: int = 5
    sparse_top_k: int = 5
    final_top_k: int = 5


class RetrievalConfigResponse(BaseModel):
    dense_top_k: int
    sparse_top_k: int
    final_top_k: int


class ModelConfigRequest(BaseModel):
    models: list[str]


class ModelConfigResponse(BaseModel):
    models: list[str]


class FeatureFlagsResponse(BaseModel):
    otel_enabled: bool
    memory_enabled: bool


class OtelToggleRequest(BaseModel):
    otel_enabled: bool


class MemoryToggleRequest(BaseModel):
    memory_enabled: bool
