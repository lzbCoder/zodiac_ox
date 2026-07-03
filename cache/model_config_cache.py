"""
线程安全的内存缓存模块——用于模型配置的热加载

embedding 调用发生在 asyncio.to_thread 线程中（sync 上下文），不能使用
async DB 查询。本模块提供线程安全的同步读接口，在保存配置时立即刷新缓存。

用法:
    from cache.model_config_cache import get_embedding_model
    model = get_embedding_model()  # 线程安全，不阻塞
"""
import time
import threading
from config import EMBEDDING_MODEL as _ENV_EMBEDDING_MODEL

_lock = threading.Lock()

# ── embedding model ──
_embedding_model: str | None = None
_embedding_ts: float = 0.0
_EMBEDDING_TTL = 600  # 兜底：10 分钟后缓存自动失效

# ── ragas defaults ──
DEFAULT_RAGAS_ANSWER = ["qwen3-max"]
DEFAULT_RAGAS_EVAL = ["qwen3.6-flash"]
_ragas_answer: list[str] | None = None
_ragas_eval: list[str] | None = None
_ragas_ts: float = 0.0
_RAGAS_TTL = 600


def get_embedding_model() -> str:
    """返回当前缓存的 embedding 模型名。可在任意线程中安全调用。"""
    with _lock:
        if _embedding_model and (time.time() - _embedding_ts) < _EMBEDDING_TTL:
            return _embedding_model
    return _ENV_EMBEDDING_MODEL


def set_embedding_model_cache(model: str) -> None:
    """保存后立即刷新 embedding 缓存（热加载）。"""
    global _embedding_model, _embedding_ts
    with _lock:
        _embedding_model = model
        _embedding_ts = time.time()


def get_ragas_default_answer_model() -> str:
    """返回 RAGAS 答案生成模型的第一个（默认值）。"""
    answer, _ = _get_ragas_defaults()
    return answer[0] if answer else DEFAULT_RAGAS_ANSWER[0]


def get_ragas_default_eval_model() -> str:
    """返回 RAGAS 评分模型的第一个（默认值）。"""
    _, eval_models = _get_ragas_defaults()
    return eval_models[0] if eval_models else DEFAULT_RAGAS_EVAL[0]


def _get_ragas_defaults() -> tuple[list[str], list[str]]:
    with _lock:
        if _ragas_answer is not None and (time.time() - _ragas_ts) < _RAGAS_TTL:
            return _ragas_answer, _ragas_eval or DEFAULT_RAGAS_EVAL
    return DEFAULT_RAGAS_ANSWER, DEFAULT_RAGAS_EVAL


def set_ragas_cache(answer_models: list[str], eval_models: list[str]) -> None:
    """保存后立即刷新 RAGAS 模型缓存（热加载）。"""
    global _ragas_answer, _ragas_eval, _ragas_ts
    with _lock:
        _ragas_answer = list(answer_models)
        _ragas_eval = list(eval_models)
        _ragas_ts = time.time()


def init_model_cache_from_db_sync() -> None:
    """启动时用 psycopg2 同步连接种子缓存。失败静默回退到 env var。"""
    import json
    try:
        import psycopg2
        from config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD

        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE,
            user=PG_USER, password=PG_PASSWORD,
            connect_timeout=5,
        )
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT config_key, config_value FROM root.system_configs "
                "WHERE config_key IN (%s, %s, %s)",
                ("embedding.model", "ragas.answer_models", "ragas.eval_models"),
            )
            rows = {r[0]: r[1] for r in cur.fetchall()}
            cur.close()
        finally:
            conn.close()

        if rows.get("embedding.model"):
            set_embedding_model_cache(rows["embedding.model"].strip())
        if rows.get("ragas.answer_models"):
            try:
                ans = json.loads(rows["ragas.answer_models"])
                evl = json.loads(rows.get("ragas.eval_models", "[]"))
                set_ragas_cache(ans, evl)
            except (ValueError, TypeError):
                pass
    except Exception:
        # 不阻塞启动 — env var fallback 已加载
        pass
