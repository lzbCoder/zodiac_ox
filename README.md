# 越群山知识库RAG系统 - 后端

基于 FastAPI + LlamaIndex + PostgreSQL + Milvus 的轻量化知识库RAG后端服务。

## 技术栈

- **Python 3.14** + **uv** 包管理
- **FastAPI** 异步 Web 框架
- **SQLAlchemy 2.0** 异步 ORM
- **LlamaIndex** 文档解析与RAG引擎
- **DashScope** LLM (qwen3-max) + Embedding (text-embedding-v4)
- **PostgreSQL** 结构化数据存储
- **Milvus 2.4+** 向量数据库

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

系统使用默认配置连接 PostgreSQL 和 Milvus，可通过环境变量覆盖:

```bash
export PG_HOST=123.56.74.230
export PG_PORT=5432
export PG_DATABASE=ox
export PG_USER=postgres
export PG_PASSWORD=postgres@zodiac
export MILVUS_URI=http://123.56.74.230:19530
export MILVUS_TOKEN=root:Milvus
export DASHSCOPE_API_KEY=sk-xxx
export LLM_MODEL=qwen3-max
export EMBEDDING_MODEL=text-embedding-v4
```

### 3. 启动服务

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000/docs 查看 API 文档。

## API 模块

| 模块 | 前缀 | 说明 |
|------|------|------|
| 知识库管理 | `/api/kb` | 创建/删除/重命名/列表 |
| 文档管理 | `/api/documents` | 上传/解析/预览/删除 |
| RAG问答 | `/api/chat` | 语义检索 + AI流式生成 |
| 对话历史 | `/api/history` | 保存/查询/导出 |
| 向量管理 | `/api/vector` | 状态查看/重建/清理 |
| 系统设置 | `/api/system` | 配置/连接检测 |

## Docker 部署

```bash
docker build -t ox-backend .
docker run -p 8000:8000 \
  -e PG_HOST=123.56.74.230 \
  -e MILVUS_URI=http://123.56.74.230:19530 \
  -e DASHSCOPE_API_KEY=sk-xxx \
  -v $(pwd)/data:/app/data \
  ox-backend
```

## 项目结构

```
ox/
├── main.py              # FastAPI 入口
├── config.py            #[chat.py](schemas/chat.py) 全局配置
├── databa[chat.py](schemas/chat.py)se.py          # PostgreSQL 连接
├── milvus_client.py  [chat.py](schemas/chat.py)   # Milvus 连接
├── models/              # SQLAlchemy ORM 模型
├── schemas/             # Pydantic 请求/响应模型
├── routers/             # API 路由层[history.py](../../../python/news-project/news-project/crud/history.py)
├── services/            # 业务逻辑层
├── data/                # 文档存储目录[history.py](../../../python/news-project/news-project/crud/history.py)
├── Dockerfile
└── pyproject.toml
```
