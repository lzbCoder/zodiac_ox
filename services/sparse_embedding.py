import jieba
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction

SPARSE_DIM = 100000


class JiebaSparseEmbedding(BaseSparseEmbeddingFunction):
    """BM25 风格稀疏嵌入，使用 jieba 分词 + 哈希维度映射。"""

    def encode_documents(self, documents: list[str]) -> list[dict[int, float]]:
        return [self._encode_one(doc) for doc in documents]

    def encode_queries(self, queries: list[str]) -> list[dict[int, float]]:
        return self._encode_batch(queries)

    async def async_encode_queries(self, queries: list[str]) -> list[dict[int, float]]:
        return self._encode_batch(queries)

    def _encode_batch(self, texts: list[str]) -> list[dict[int, float]]:
        return [self._encode_one(t) for t in texts]

    @staticmethod
    def _encode_one(text: str) -> dict[int, float]:
        tokens = list(jieba.cut(text))
        tf: dict[int, int] = {}
        for t in tokens:
            dim = hash(t) % SPARSE_DIM
            tf[dim] = tf.get(dim, 0) + 1
        if not tf:
            return {}
        max_tf = max(tf.values())
        return {k: v / max_tf for k, v in tf.items()}
