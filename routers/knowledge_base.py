from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from schemas.knowledge_base import KBCreate, KBUpdate, KBResponse
from services import kb_service

router = APIRouter(prefix="/api/kb", tags=["知识库管理"])


@router.post("", response_model=KBResponse)
async def create_kb(data: KBCreate, db: AsyncSession = Depends(get_db)):
    return await kb_service.create_kb(db, data.name, data.description)


@router.get("", response_model=list[KBResponse])
async def list_kbs(db: AsyncSession = Depends(get_db)):
    return await kb_service.list_kbs(db)


@router.get("/{kb_id}", response_model=KBResponse)
async def get_kb(kb_id: int, db: AsyncSession = Depends(get_db)):
    kb = await kb_service.get_kb(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return kb


@router.put("/{kb_id}", response_model=KBResponse)
async def update_kb(kb_id: int, data: KBUpdate, db: AsyncSession = Depends(get_db)):
    kb = await kb_service.update_kb(db, kb_id, data.name, data.description)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return kb


@router.delete("/{kb_id}")
async def delete_kb(kb_id: int, db: AsyncSession = Depends(get_db)):
    ok = await kb_service.delete_kb(db, kb_id)
    if not ok:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return {"message": "已删除"}
