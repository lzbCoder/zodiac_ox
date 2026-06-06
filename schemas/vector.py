from pydantic import BaseModel


class VectorStats(BaseModel):
    total_vectors: int
    valid_vectors: int
    doc_count: int
    collection_name: str
