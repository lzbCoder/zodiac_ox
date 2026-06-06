import os
import sys
import gzip
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from loguru import logger

TZ_CN = timezone(timedelta(hours=8))
ROTATE_SIZE = "1 MB"


class InterceptHandler(logging.Handler):
    """Redirect standard logging records to Loguru."""

    def emit(self, record: logging.LogRecord):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _compress_and_rename(path: str) -> None:
    """loguru compression hook：将滚动后的日志压缩归档到 archive/YYYY-MM/ 目录。

    loguru 传入的 path 是已被重命名的旧日志文件路径
    （如 logs/app.log.2026-05-23T14-30-00.000）。

    归档路径示例：
      logs/archive/2026-05/app_20260523_1.log.gz
      logs/archive/2026-05/error_20260523_1.log.gz
    """
    logs_dir = os.path.dirname(path)
    now = datetime.now(TZ_CN)
    date_prefix = now.strftime("%Y%m%d")
    archive_dir = os.path.join(logs_dir, "archive", now.strftime("%Y-%m"))
    os.makedirs(archive_dir, exist_ok=True)

    old_name = os.path.basename(path)
    prefix = old_name.split(".")[0]  # "app" 或 "error"

    n = 1
    while os.path.exists(os.path.join(archive_dir, f"{prefix}_{date_prefix}_{n}.log.gz")):
        n += 1

    archive_path = os.path.join(archive_dir, f"{prefix}_{date_prefix}_{n}.log.gz")

    with open(path, "rb") as f_in:
        with gzip.open(archive_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(path)

    _cleanup_old_archives(logs_dir, months=3)


def _cleanup_old_archives(logs_dir: str, months: int = 3) -> None:
    """删除超过指定月数的归档目录（基于目录名 YYYY-MM 判断）。"""
    archive_root = os.path.join(logs_dir, "archive")
    if not os.path.isdir(archive_root):
        return
    now = datetime.now(TZ_CN)
    cutoff_year = now.year
    cutoff_month = now.month - months
    while cutoff_month <= 0:
        cutoff_month += 12
        cutoff_year -= 1

    for entry in os.listdir(archive_root):
        entry_path = os.path.join(archive_root, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            y, m = entry.split("-")
            if (int(y), int(m)) < (cutoff_year, cutoff_month):
                shutil.rmtree(entry_path)
        except (ValueError, OSError):
            pass


# Minimal no-op logging config — prevents Uvicorn from installing its own
# handlers/levels, preserving the Loguru InterceptHandler we set directly.
UVICORN_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {},
    "handlers": {},
    "loggers": {},
}


def setup_logger():
    """Initialize Loguru as the unified logging system.

    Configures console and file outputs, takes over Uvicorn and standard
    library logging so all logs flow through Loguru.
    """
    logger.remove()

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # --- Console output: INFO and above ---
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | <level>{level: <8}</level> | {message}",
        level="INFO",
        colorize=True,
    )

    # --- App-level file log: INFO, size rotation, archive to YYYY-MM/ ---
    logger.add(
        logs_dir / "app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | <level>{level: <8}</level> | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation=ROTATE_SIZE,
        compression=_compress_and_rename,
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    # --- Error-only file log: ERROR, size rotation, archive to YYYY-MM/ ---
    logger.add(
        logs_dir / "error.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | <level>{level: <8}</level> | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation=ROTATE_SIZE,
        compression=_compress_and_rename,
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    # --- Take over standard library logging ---
    root = logging.getLogger()
    root.handlers = [InterceptHandler()]
    root.setLevel(logging.INFO)

    # Route Uvicorn startup/error logs through Loguru
    for name in ["uvicorn", "uvicorn.error", "uvicorn.asgi"]:
        ulog = logging.getLogger(name)
        ulog.handlers = [InterceptHandler()]
        ulog.propagate = False
        ulog.setLevel(logging.INFO)

    # Uvicorn access logs: still intercepted but set to WARNING so
    # they don't duplicate our HTTP middleware access logs
    access_log = logging.getLogger("uvicorn.access")
    access_log.handlers = [InterceptHandler()]
    access_log.propagate = False
    access_log.setLevel(logging.WARNING)

    # Override the uvicorn-internal LOGGING_CONFIG so uvicorn
    # does not re-install its own handlers on startup
    import uvicorn.config
    uvicorn.config.LOGGING_CONFIG = UVICORN_LOGGING_CONFIG

    return logger
