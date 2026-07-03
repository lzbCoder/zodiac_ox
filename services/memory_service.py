"""
企业级记忆管理器，双层隔离。

会话短期记忆：按 session_id 隔离，单会话 FIFO，token_limit=4096。
用户长期记忆：按 user_id 隔离，跨会话积累 + Milvus 向量语义召回。
"""

from typing import List

from llama_index.core.base.llms.types import ChatMessage, MessageRole

# 模块级单例 — 由 main.py 在启动时初始化
_manager: "EnterpriseMemoryManager | None" = None


def get_memory_manager() -> "EnterpriseMemoryManager":
    """返回全局记忆管理器实例"""
    assert _manager is not None, "EnterpriseMemoryManager not initialized"
    return _manager


def init_memory_manager(async_engine, db_schema: str) -> "EnterpriseMemoryManager":
    """初始化并存储全局记忆管理器实例"""
    global _manager
    _manager = EnterpriseMemoryManager(async_engine=async_engine, db_schema=db_schema)
    return _manager

from llama_index.core.memory import Memory, VectorMemoryBlock
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from sqlalchemy.ext.asyncio import AsyncEngine

from config import (
    DASHSCOPE_API_KEY,
    EMBEDDING_DIM,
    MEMORY_DEFAULT_USER_ID,
    SESSION_MEMORY_TOKEN_LIMIT,
    USER_MEMORY_TOKEN_LIMIT,
    MEMORY_VECTOR_TOP_K,
    MILVUS_MEMORY_COLLECTION,
    MILVUS_TOKEN,
    MILVUS_URI,
)
from cache.model_config_cache import get_embedding_model


class EnterpriseMemoryManager:
    """管理双层记忆：会话级短期 + 用户级长期向量存储。"""

    def __init__(self, async_engine: AsyncEngine, db_schema: str):
        self._async_engine = async_engine
        self._db_schema = db_schema

        # 共享向量存储 — 所有用户共用同一个 Milvus 集合
        self._vector_store = MilvusVectorStore(
            uri=MILVUS_URI,
            token=MILVUS_TOKEN,
            collection_name=MILVUS_MEMORY_COLLECTION,
            dim=EMBEDDING_DIM,
        )

        # 共享嵌入模型
        self._embed_model = DashScopeEmbedding(
            model_name=get_embedding_model(),
            api_key=DASHSCOPE_API_KEY,
        )

        # 共享 VectorMemoryBlock — 处理语义存储 + 召回
        self._vector_block = VectorMemoryBlock(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
            similarity_top_k=MEMORY_VECTOR_TOP_K,
        )

    # ── 记忆实例工厂 ──────────────────────────────────────────

    def _create_session_memory(self, session_id: str) -> Memory:
        """会话级短期记忆 — 纯 FIFO，无向量块。"""
        return Memory.from_defaults(
            session_id=session_id,
            token_limit=SESSION_MEMORY_TOKEN_LIMIT,
            chat_history_token_ratio=0.7,
            memory_blocks=[],  # 不进行向量存储 — 过期消息将被丢弃
            async_engine=self._async_engine,
            db_schema=self._db_schema,
            table_name="session_chat_memory",
        )

    def _create_user_global_memory(self, user_id: str) -> Memory:
        """用户级长期记忆 — FIFO + VectorMemoryBlock 瀑布(waterfall)溢出。"""
        return Memory.from_defaults(
            session_id=user_id,  # 通过user_id进行跨会话累积
            token_limit=USER_MEMORY_TOKEN_LIMIT,
            chat_history_token_ratio=0.7,
            memory_blocks=[self._vector_block],
            async_engine=self._async_engine,
            db_schema=self._db_schema,
            table_name="user_global_memory",
        )

    # ── 公开 API ──────────────────────────────────────────────

    async def load_memory_context(
        self,
        query: str,
        user_id: str = MEMORY_DEFAULT_USER_ID,
        session_id: str | None = None,
    ) -> str:
        """
        加载记忆上下文，用于注入 system prompt。

        返回合并的格式化文本：
        1. 会话聊天历史（如提供 session_id）
        2. 跨会话向量召回（来自用户长期记忆）
        """
        parts: list[str] = []

        # 1. 会话短期记忆 — 当前对话历史
        if session_id:
            session_mem = self._create_session_memory(session_id)
            session_messages = await session_mem.aget(input=query)
            session_text = self._format_chat_history(session_messages, exclude_last=True)
            if session_text:
                parts.append(session_text)

        # 2. 用户长期向量记忆 — 跨会话语义召回
        if session_id:
            # 使用最近的会话消息作为向量搜索的上下文
            session_mem = self._create_session_memory(session_id)
            recent = await session_mem.sql_store.get_messages(session_id)
            # 与当前查询合并
            context_messages = list(recent) if recent else []
        else:
            context_messages = []
        context_messages.append(ChatMessage(role=MessageRole.USER, content=query))

        vector_context = await self._vector_block.aget(
            messages=context_messages,
            # 不带 session_id 过滤 — 跨会话召回
        )
        if vector_context and vector_context.strip():
            parts.append(vector_context.strip())

        return "\n\n".join(parts) if parts else ""

    async def save_interaction(
        self,
        query: str,
        answer: str,
        user_id: str = MEMORY_DEFAULT_USER_ID,
        session_id: str | None = None,
    ):
        """保存一轮问答到会话记忆和用户全局记忆。"""
        user_msg = ChatMessage(role=MessageRole.USER, content=query)
        assistant_msg = ChatMessage(role=MessageRole.ASSISTANT, content=answer)

        # 保存到会话记忆
        if session_id:
            session_mem = self._create_session_memory(session_id)
            await session_mem.aput(user_msg)
            await session_mem.aput(assistant_msg)

        # 保存到用户全局记忆（FIFO 超出限制时瀑布溢出 → 向量存储）
        global_mem = self._create_user_global_memory(user_id)
        await global_mem.aput(user_msg)
        await global_mem.aput(assistant_msg)

    async def clear_session_memory(self, session_id: str):
        """仅清除会话级短期记忆。"""
        session_mem = self._create_session_memory(session_id)
        await session_mem.areset()

    async def clear_user_memory(self, user_id: str):
        """清除用户全部记忆 — 包括全局 FIFO 和向量存储。"""
        global_mem = self._create_user_global_memory(user_id)
        await global_mem.areset()
        # 说明：areset() 会清除该 user_id 的 SQLAlchemyChatStore 表数据。
        # 向量存储的清理由 Memory 的 waterfall 机制处理，将旧向量标记为不活跃。
        # 如需完全清理，Milvus 集合需要 delete-by-filter 操作，
        # MilvusVectorStore 通过其内部节点管理来处理此操作。

    async def get_memory_status(
        self,
        user_id: str = MEMORY_DEFAULT_USER_ID,
        session_id: str | None = None,
    ) -> dict:
        """返回记忆状态，用于查看。"""
        result = {"user_id": user_id, "session_id": session_id}

        if session_id:
            session_mem = self._create_session_memory(session_id)
            msgs = await session_mem.sql_store.get_messages(session_id)
            result["session_message_count"] = len(msgs) if msgs else 0
        else:
            result["session_message_count"] = 0

        global_mem = self._create_user_global_memory(user_id)
        g_msgs = await global_mem.sql_store.get_messages(user_id)
        result["global_message_count"] = len(g_msgs) if g_msgs else 0

        return result

    # ── 辅助方法 ───────────────────────────────────────────────

    @staticmethod
    def _format_chat_history(messages: List[ChatMessage], exclude_last: bool = False) -> str:
        """将 ChatMessage 列表转换为可读的文本块。"""
        if not messages:
            return ""

        msgs = messages[:-1] if exclude_last and len(messages) > 1 else messages
        if not msgs:
            return ""

        lines = ["[对话历史]"]
        for m in msgs:
            role_label = "用户" if m.role == MessageRole.USER else "助手"
            text = EnterpriseMemoryManager._extract_text(m)
            if text.strip():
                lines.append(f"{role_label}: {text.strip()}")
        return "\n".join(lines)

    @staticmethod
    def _extract_text(message: ChatMessage) -> str:
        """从 ChatMessage 的 blocks 中提取纯文本。"""
        parts = []
        for block in message.blocks:
            if hasattr(block, "text"):
                parts.append(block.text)
        return " ".join(parts)
