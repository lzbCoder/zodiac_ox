import redis.asyncio as redis
from config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB
from loguru import logger
_redis: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
        )
        await _redis.ping()
    logger.info("Redis 连接成功")
    return _redis


async def close_redis():
    global _redis
    if _redis:
        await _redis.close()
        _redis = None
