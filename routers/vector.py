from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from schemas.vector import VectorStats
from services import vector_service
from milvus_client import get_collection

router = APIRouter(prefix="/api/vector", tags=["向量管理"])


@router.get("/stats", response_model=VectorStats)
async def get_stats(db: AsyncSession = Depends(get_db)):
    collection = get_collection()
    stats = await vector_service.get_vector_stats(collection, db)
    return stats


@router.post("/cleanup")
async def clean_invalid_vectors(db: AsyncSession = Depends(get_db)):
    collection = get_collection()
    await vector_service.clean_invalid_vectors(collection, db)
    return {"message": "清理完成"}
