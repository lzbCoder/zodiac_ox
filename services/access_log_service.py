"""客户端访问记录服务。

职责：
1. 从请求中解析真实客户端 IP（兼容反向代理 X-Forwarded-For / X-Real-IP）。
2. 按 session_id（无则 client_ip）在 30 分钟窗口内首次去重，避免每次接口调用刷库。
   去重基于 Redis SET NX EX；Redis 不可用时「失败即记录」，审计完整性优先。
3. 写入 root.access_log 表。

设计原则：绝不影响主流程——所有对外入口均包 try/except，记录失败只告警。
"""
from datetime import datetime
from loguru import logger
from starlette.requests import Request

from database import async_session
from redis_client import get_redis
from models.access_log import AccessLog

# 同一访客（session 或 IP）在该时间窗口内只记录一次
DEDUP_WINDOW_SECONDS = 30 * 60
_DEDUP_PREFIX = "access_log:dedup:"


def get_client_ip(request: Request) -> str | None:
    """解析真实客户端 IP，兼容反向代理部署。

    优先级：X-Forwarded-For 首个地址 > X-Real-IP > 直连 socket 地址。
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        # XFF 形如 "client, proxy1, proxy2"，第一个为真实客户端
        return xff.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    if request.client:
        return request.client.host
    return None


def _extract_identity(request: Request) -> tuple[str | None, str | None]:
    """尽力从 query 参数 / 自定义头提取 session_id、user_id。

    业务接口的 session_id/user_id 多在请求体内，中间件不解析 body，
    因此这里只取 query 参数与 X-Session-Id / X-User-Id 头，取不到则为 None。
    """
    session_id = request.query_params.get("session_id") or request.headers.get("x-session-id")
    user_id = request.query_params.get("user_id") or request.headers.get("x-user-id")
    return session_id, user_id


async def _is_first_visit(dedup_value: str) -> bool:
    """Redis SET NX EX 去重判定。

    返回 True 表示窗口内首次（应记录）。Redis 不可用时返回 True，
    即「失败即记录」，保证审计完整性。
    """
    try:
        redis = await get_redis()
        # NX：键不存在才设置；设置成功 => 窗口内首次访问
        created = await redis.set(
            f"{_DEDUP_PREFIX}{dedup_value}", "1", nx=True, ex=DEDUP_WINDOW_SECONDS
        )
        return bool(created)
    except Exception as e:
        logger.warning(f"访问记录去重 Redis 不可用，按首次处理：{e}")
        return True


async def _persist(record: dict) -> None:
    """写入 access_log 表。失败只告警，不抛出。"""
    try:
        async with async_session() as db:
            db.add(AccessLog(**record))
            await db.commit()
    except Exception as e:
        logger.warning(f"访问记录落库失败：{e}")


async def record_access(request: Request, status_code: int, cost_ms: int) -> None:
    """中间件入口：写访问日志文件 + 按需落库。全程异常隔离。"""
    try:
        client_ip = get_client_ip(request)
        session_id, user_id = _extract_identity(request)
        method = request.method
        path = request.url.path
        user_agent = request.headers.get("user-agent")
        referer = request.headers.get("referer")

        # 1) 写日志文件（loguru，始终记录）
        logger.info(
            f"ACCESS ip={client_ip} session={session_id} user={user_id} "
            f"{method} {path} -> {status_code} ({cost_ms}ms) ua={user_agent}"
        )

        # 2) 按 session/IP 首次去重后落库
        dedup_value = session_id or client_ip
        if not dedup_value:
            return
        if not await _is_first_visit(dedup_value):
            return

        await _persist({
            "client_ip": client_ip,
            "session_id": session_id,
            "user_id": user_id,
            "method": method,
            "path": path[:500] if path else None,
            "status_code": status_code,
            "user_agent": user_agent,
            "referer": referer[:500] if referer else None,
            "cost_ms": cost_ms,
            "create_time": datetime.now(),
        })
    except Exception as e:
        # 兜底：访问记录绝不影响主流程
        logger.warning(f"访问记录处理异常：{e}")
