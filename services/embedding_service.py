import asyncio
import uuid
from typing import TYPE_CHECKING
from dashscope import TextEmbedding
from config import DASHSCOPE_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM
from services.sparse_embedding import JiebaSparseEmbedding

if TYPE_CHECKING:
    from pymilvus import Collection

_sparse_encoder = JiebaSparseEmbedding()
BATCH_SIZE = 25  # DashScope text-embedding-v4 单次最大输入条数


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """对文本列表生成向量。批量调用 DashScope API（25条/次），卸载到线程池避免阻塞事件循环。"""
    return await asyncio.to_thread(_embed_texts_sync, texts)


def _embed_texts_sync(texts: list[str]) -> list[list[float]]:
    result: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        resp = TextEmbedding.call(
            api_key=DASHSCOPE_API_KEY,
            model=EMBEDDING_MODEL,
            input=[t[:2048] for t in batch],
            dimensions=EMBEDDING_DIM,
        )
        if resp.status_code == 200:
            for emb in resp.output["embeddings"]:
                result.append(emb["embedding"])
        else:
            result.extend([[0.0] * EMBEDDING_DIM] * len(batch))
    return result


async def insert_vectors(
    collection: "Collection", kb_id: int, doc_id: int,
    chunks: list[dict], embeddings: list[list[float]],
) -> list[str]:
    """异步包装：在 thread pool 中执行 Milvus insert（避免阻塞事件循环）。"""
    return await asyncio.to_thread(_insert_vectors_sync, collection, kb_id, doc_id, chunks, embeddings)


def _insert_vectors_sync(
    collection: "Collection", kb_id: int, doc_id: int,
    chunks: list[dict], embeddings: list[list[float]],
) -> list[str]:
    contents = [c["content"] for c in chunks]
    sparse_vectors = _sparse_encoder.encode_documents(contents)

    milvus_ids = []
    entities = []
    for chunk, emb, sparse in zip(chunks, embeddings, sparse_vectors):
        mid = uuid.uuid4().hex[:16]
        milvus_ids.append(mid)
        entities.append({
            "id": mid,
            "dense_vector": emb,
            "sparse_vector": sparse,
            "kb_id": kb_id,
            "doc_id": doc_id,
            "chunk_id": chunk["id"],
        })
    collection.insert(entities)
    # 不再手动 flush — Milvus 自动定期落盘，手动 flush 阻塞写入线程
    return milvus_ids


def delete_vectors_by_doc(collection: "Collection", doc_id: int):
    collection.delete(f'doc_id == {doc_id}')


def delete_vectors_by_kb(collection: "Collection", kb_id: int):
    collection.delete(f'kb_id == {kb_id}')
