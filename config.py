import os
from pathlib import Path
from urllib.parse import quote_plus
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# PostgreSQL
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DATABASE = os.getenv("PG_DATABASE", "ox")
PG_SCHEMA = os.getenv("PG_SCHEMA", "root")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD")
if not PG_PASSWORD:
    raise ValueError("PG_PASSWORD 未设置，请在 .env 文件中配置")

_pg_pass = quote_plus(PG_PASSWORD)
DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{_pg_pass}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
DATABASE_URL_SYNC = f"postgresql+psycopg2://{PG_USER}:{_pg_pass}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# Milvus
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
if not MILVUS_TOKEN:
    raise ValueError("MILVUS_TOKEN 未设置，请在 .env 文件中配置")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "local_document_vectors")

# DashScope
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY 未设置，请在 .env 文件中配置")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
if not REDIS_PASSWORD:
    raise ValueError("REDIS_PASSWORD 未设置，请在 .env 文件中配置")
REDIS_DB = int(os.getenv("REDIS_DB", "1"))

# LangSmith / OpenTelemetry
# Set OTEL_SERVICE_NAME early so the SDK auto-detects it as a fallback.
os.environ.setdefault("OTEL_SERVICE_NAME", "rag-system")

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
if not LANGCHAIN_API_KEY:
    raise ValueError("LANGCHAIN_API_KEY 未设置，请在 .env 文件中配置")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com/otel/v1/traces")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-system")

# Memory (LlamaIndex Memory API)
SESSION_MEMORY_TOKEN_LIMIT = int(os.getenv("SESSION_MEMORY_TOKEN_LIMIT", "4096"))
USER_MEMORY_TOKEN_LIMIT = int(os.getenv("USER_MEMORY_TOKEN_LIMIT", "100000"))
MEMORY_VECTOR_TOP_K = int(os.getenv("MEMORY_VECTOR_TOP_K", "3"))
MILVUS_MEMORY_COLLECTION = os.getenv("MILVUS_MEMORY_COLLECTION", "local_rag_user_memory")
MEMORY_DEFAULT_USER_ID = os.getenv("MEMORY_DEFAULT_USER_ID", "admin")

# Default chunk config
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_SPLIT_SEPARATOR = "\n\n"
