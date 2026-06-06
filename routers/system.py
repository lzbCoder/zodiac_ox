from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db, engine
from schemas.system import SystemConfigCreate, SystemConfigUpdate, SystemConfigResponse, ConnectionStatus, PromptConfigRequest, PromptConfigResponse, RetrievalConfigRequest, RetrievalConfigResponse, FeatureFlagsResponse, OtelToggleRequest, MemoryToggleRequest
from models.system_config import SystemConfig
from milvus_client import get_collection, reinit_collection
from redis_client import get_redis
from pymilvus import utility
from config import MILVUS_COLLECTION

router = APIRouter(prefix="/api/system", tags=["系统设置"])


@router.get("/configs", response_model=list[SystemConfigResponse])
async def list_configs(db: AsyncSession = Depends(get_db)):
    stmt = select(SystemConfig).order_by(SystemConfig.config_key)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/configs/{config_key}", response_model=SystemConfigResponse)
async def get_config(config_key: str, db: AsyncSession = Depends(get_db)):
    stmt = select(SystemConfig).where(SystemConfig.config_key == config_key)
    result = await db.execute(stmt)
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="配置不存在")
    return config


@router.post("/configs", response_model=SystemConfigResponse)
async def create_config(data: SystemConfigCreate, db: AsyncSession = Depends(get_db)):
    config = SystemConfig(config_key=data.config_key, config_value=data.config_value, description=data.description)
    db.add(config)
    await db.commit()
    await db.refresh(config)
    return config


@router.put("/configs/{config_key}", response_model=SystemConfigResponse)
async def update_config(config_key: str, data: SystemConfigUpdate, db: AsyncSession = Depends(get_db)):
    stmt = select(SystemConfig).where(SystemConfig.config_key == config_key)
    result = await db.execute(stmt)
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="配置不存在")
    config.config_value = data.config_value
    if data.description is not None:
        config.description = data.description
    await db.commit()
    await db.refresh(config)
    return config


@router.delete("/configs/{config_key}")
async def delete_config(config_key: str, db: AsyncSession = Depends(get_db)):
    stmt = select(SystemConfig).where(SystemConfig.config_key == config_key)
    result = await db.execute(stmt)
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="配置不存在")
    await db.delete(config)
    await db.commit()
    return {"message": "已删除"}


@router.get("/connection-status", response_model=ConnectionStatus)
async def check_connections():
    pg_ok = False
    milvus_ok = False
    redis_ok = False
    try:
        async with engine.connect() as conn:
            await conn.execute(select(1))
        pg_ok = True
    except Exception:
        pass
    try:
        milvus_ok = utility.has_collection(MILVUS_COLLECTION)
    except Exception:
        pass
    try:
        r = await get_redis()
        await r.ping()
        redis_ok = True
    except Exception:
        pass
    return {"postgresql": pg_ok, "milvus": milvus_ok, "redis": redis_ok}


@router.get("/default-chunk-config")
async def get_default_chunk_config(db: AsyncSession = Depends(get_db)):
    """从 system_configs 表返回全局默认 chunk 配置。"""
    stmt = select(SystemConfig).where(SystemConfig.config_key.in_([
        "default_chunk_size", "default_chunk_overlap", "default_split_separator"
    ]))
    result = await db.execute(stmt)
    rows = {r.config_key: r.config_value for r in result.scalars().all()}
    return {
        "chunk_size": int(rows.get("default_chunk_size") or 1000),
        "chunk_overlap": int(rows.get("default_chunk_overlap") or 100),
        "split_separator": rows.get("default_split_separator") or "\n\n",
    }


DEFAULT_SYSTEM_PROMPT = (
    "你是一个知识库问答助手。请根据提供的文档片段回答用户问题。"
    "如果文档片段不足以回答问题，请如实说明。回答时请引用具体的来源。"
)
DEFAULT_USER_PROMPT = "文档片段：\n{context}\n\n用户问题：{query}\n\n请根据以上文档片段回答问题："


@router.get("/prompt-config", response_model=PromptConfigResponse)
async def get_prompt_config(db: AsyncSession = Depends(get_db)):
    stmt = select(SystemConfig).where(SystemConfig.config_key.in_([
        "system_prompt", "user_prompt"
    ]))
    result = await db.execute(stmt)
    rows = {r.config_key: r.config_value for r in result.scalars().all()}
    return {
        "system_prompt": rows.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        "user_prompt": rows.get("user_prompt") or DEFAULT_USER_PROMPT,
    }


@router.post("/prompt-config")
async def save_prompt_config(data: PromptConfigRequest, db: AsyncSession = Depends(get_db)):
    for key, value in [("system_prompt", data.system_prompt), ("user_prompt", data.user_prompt)]:
        stmt = select(SystemConfig).where(SystemConfig.config_key == key)
        result = await db.execute(stmt)
        config = result.scalar_one_or_none()
        if config:
            config.config_value = value
        else:
            db.add(SystemConfig(config_key=key, config_value=value, description=f"{key}配置"))
    await db.commit()
    return {"message": "配置保存成功"}


DEFAULT_RETRIEVAL_CONFIG = {"dense_top_k": 5, "sparse_top_k": 5, "final_top_k": 5}


@router.get("/retrieval-config", response_model=RetrievalConfigResponse)
async def get_retrieval_config(db: AsyncSession = Depends(get_db)):
    stmt = select(SystemConfig).where(SystemConfig.config_key.in_([
        "retrieval.dense_top_k", "retrieval.sparse_top_k", "retrieval.final_top_k"
    ]))
    result = await db.execute(stmt)
    rows = {r.config_key: r.config_value for r in result.scalars().all()}
    return {
        "dense_top_k": int(rows.get("retrieval.dense_top_k") or DEFAULT_RETRIEVAL_CONFIG["dense_top_k"]),
        "sparse_top_k": int(rows.get("retrieval.sparse_top_k") or DEFAULT_RETRIEVAL_CONFIG["sparse_top_k"]),
        "final_top_k": int(rows.get("retrieval.final_top_k") or DEFAULT_RETRIEVAL_CONFIG["final_top_k"]),
    }


@router.post("/retrieval-config")
async def save_retrieval_config(data: RetrievalConfigRequest, db: AsyncSession = Depends(get_db)):
    if not (1 <= data.dense_top_k <= 100):
        raise HTTPException(status_code=400, detail="dense_top_k 必须在 1-100 之间")
    if not (1 <= data.sparse_top_k <= 100):
        raise HTTPException(status_code=400, detail="sparse_top_k 必须在 1-100 之间")
    if not (1 <= data.final_top_k <= 100):
        raise HTTPException(status_code=400, detail="final_top_k 必须在 1-100 之间")
    if data.final_top_k > max(data.dense_top_k, data.sparse_top_k):
        raise HTTPException(status_code=400, detail="final_top_k 不能超过 dense_top_k 和 sparse_top_k 的最大值")

    for key, value in [
        ("retrieval.dense_top_k", str(data.dense_top_k)),
        ("retrieval.sparse_top_k", str(data.sparse_top_k)),
        ("retrieval.final_top_k", str(data.final_top_k)),
    ]:
        stmt = select(SystemConfig).where(SystemConfig.config_key == key)
        result = await db.execute(stmt)
        config = result.scalar_one_or_none()
        if config:
            config.config_value = value
        else:
            db.add(SystemConfig(config_key=key, config_value=value, description=f"检索参数-{key}"))
    await db.commit()
    return {"message": "检索参数配置保存成功"}


@router.post("/reinit-collection")
async def reinit_vector_collection():
    """删除并使用最新 schema 重建 Milvus 集合。所有向量将丢失，需重新导入。"""
    try:
        reinit_collection()
        return {"message": "向量库已重建，请重新上传文档以生成向量"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建向量库失败: {str(e)}")


def _read_bool_config(rows: dict, key: str, default: bool = True) -> bool:
    """从查询结果中读取布尔型配置项"""
    row = rows.get(key)
    if row is None:
        return default
    return row.config_value.lower() in ("true", "1", "yes")


@router.get("/feature-flags", response_model=FeatureFlagsResponse)
async def get_feature_flags(db: AsyncSession = Depends(get_db)):
    stmt = select(SystemConfig).where(SystemConfig.config_key.in_([
        "otel.enabled", "memory.enabled",
    ]))
    result = await db.execute(stmt)
    rows = {r.config_key: r for r in result.scalars().all()}
    return FeatureFlagsResponse(
        otel_enabled=_read_bool_config(rows, "otel.enabled", True),
        memory_enabled=_read_bool_config(rows, "memory.enabled", True),
    )


@router.post("/feature-flags/otel")
async def toggle_otel(data: OtelToggleRequest, db: AsyncSession = Depends(get_db)):
    """动态启用/禁用 OTel 监测，立即生效。"""
    from otel_tracer import set_otel_enabled

    value_str = "true" if data.otel_enabled else "false"

    stmt = select(SystemConfig).where(SystemConfig.config_key == "otel.enabled")
    result = await db.execute(stmt)
    config = result.scalar_one_or_none()
    if config:
        config.config_value = value_str
    else:
        db.add(SystemConfig(config_key="otel.enabled", config_value=value_str, description="是否启用 OTel 监测"))
    await db.commit()

    # 立即应用到内存状态
    set_otel_enabled(data.otel_enabled)

    status = "已开启" if data.otel_enabled else "已关闭"
    return {"message": f"OTel 监测{status}", "otel_enabled": data.otel_enabled}


@router.post("/feature-flags/memory")
async def toggle_memory(data: MemoryToggleRequest, db: AsyncSession = Depends(get_db)):
    """动态启用/禁用短期和长期记忆，立即生效。"""
    value_str = "true" if data.memory_enabled else "false"

    stmt = select(SystemConfig).where(SystemConfig.config_key == "memory.enabled")
    result = await db.execute(stmt)
    config = result.scalar_one_or_none()
    if config:
        config.config_value = value_str
    else:
        db.add(SystemConfig(config_key="memory.enabled", config_value=value_str, description="是否启用短期/长期记忆"))
    await db.commit()

    status = "已开启" if data.memory_enabled else "已关闭"
    return {"message": f"短期/长期记忆{status}", "memory_enabled": data.memory_enabled}
