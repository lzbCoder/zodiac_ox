"""DashScope Embedding 适配 ragas BaseRagasEmbedding 接口"""
import asyncio
from typing import Any
from dashscope import TextEmbedding
from config import DASHSCOPE_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from ragas.embeddings.base import BaseRagasEmbedding

BATCH_SIZE = EMBEDDING_BATCH_SIZE


class DashScopeRagasEmbedding(BaseRagasEmbedding):
    """将 DashScope TextEmbedding 包装为 ragas 兼容的 embedding 接口"""

    async def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        embeddings = await self.embed_texts([text], **kwargs)
        return embeddings[0]

    async def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_texts_sync, texts)

    def _embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        # Batch into groups of BATCH_SIZE to avoid API limits
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            resp = TextEmbedding.call(
                api_key=DASHSCOPE_API_KEY,
                model=EMBEDDING_MODEL,
                input=[t[:2048] for t in batch],
            )
            if resp.status_code == 200:
                for emb in resp.output["embeddings"]:
                    result.append(emb["embedding"])
            else:
                raise RuntimeError(
                    f"Embedding API failed (status={resp.status_code}) for batch starting at index {i}"
                )
        return result

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await self.embed_text(text, **kwargs)

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return await self.embed_texts(texts, **kwargs)
