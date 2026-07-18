import asyncio
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    SentenceSplitter,
    MarkdownNodeParser,
    HTMLNodeParser,
    CodeSplitter,
    JSONNodeParser,
)
from loguru import logger
import config


# 图片类扩展名列表，用于注入 EasyOCR 读取器
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def _build_reader(file_path: str) -> SimpleDirectoryReader:
    """构建 SimpleDirectoryReader，对图片文件注入 EasyOCR 读取器。"""
    ext = Path(file_path).suffix.lower()
    if ext in _IMAGE_EXTS:
        from services.ocr_reader import EasyOCRReader

        return SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor={ext: EasyOCRReader()},
        )
    return SimpleDirectoryReader(input_files=[file_path])


def _sanitize_text(text: str) -> str:
    """移除会导致 PostgreSQL UTF-8 错误的空字节和代理字符。"""
    return text.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8")


def _is_garbage(text: str) -> bool:
    """如果文本大部分是二进制噪音（控制字符、替换符号），返回 True。"""
    if len(text) < 20:
        return False
    # 统计可打印/语义字符 vs 控制/替换字符
    printable = sum(1 for c in text if c.isalnum() or c in " \n\r\t.,;:!?()[]{}<>/\\-+=@#$%^&*_|~'\"")
    control = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    replacements = text.count("�")  # Unicode replacement character
    ratio = printable / max(len(text), 1)
    # 启发式判断：可打印字符 < 30%，或大量控制字符/替换符号，则视为二进制
    return ratio < 0.30 or control > len(text) * 0.20 or replacements > 5


def _calc_gibberish_ratio(text: str) -> float:
    """计算文本的乱码率：替换字符 + 控制字符 / 总长度。"""
    if len(text) < 20:
        return 0.0
    replacements = text.count("�")
    control = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    return (replacements + control) / max(len(text), 1)


def _sentence_parser(chunk_size: int, chunk_overlap: int, split_separator: str, **__) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator=split_separator,
    )


def _markdown_parser(**__) -> MarkdownNodeParser:
    return MarkdownNodeParser()


def _html_parser(**__) -> HTMLNodeParser:
    return HTMLNodeParser()


def _json_parser(**__) -> JSONNodeParser:
    """JSON 解析器 — 按 JSON 结构拆解为扁平 key-path 文本节点。"""
    return JSONNodeParser()


def _code_parser_factory(language: str):
    """返回一个工厂函数，创建 CodeSplitter，失败时回退到 SentenceSplitter。"""
    def build(chunk_size: int = 1000, chunk_overlap: int = 100, split_separator: str = "\n\n", **__):
        try:
            return CodeSplitter(
                language=language,
                chunk_lines=max(10, chunk_size // 30),
                chunk_lines_overlap=max(1, chunk_overlap // 30),
            )
        except Exception:
            return SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                paragraph_separator=split_separator,
            )
    return build


# 映射表：文件扩展名 → 解析器工厂
# 每个工厂接收 (chunk_size, chunk_overlap, split_separator) 作为关键字参数
SPLITTER_MAP: dict[str, callable] = {
    # 结构化文档类型 — 按文档结构切分，忽略 chunk 配置
    ".md":      _markdown_parser,
    ".markdown": _markdown_parser,
    ".html":    _html_parser,
    ".htm":     _html_parser,
    # JSON 类型 — 按 JSON 结构拆解为扁平 key-path 文本
    ".json":    _json_parser,
    # 代码类型 — 按 AST 感知切分
    ".py":   _code_parser_factory("python"),
    ".java": _code_parser_factory("java"),
    ".js":   _code_parser_factory("javascript"),
    ".mjs":  _code_parser_factory("javascript"),
    ".cjs":  _code_parser_factory("javascript"),
    ".ts":   _code_parser_factory("typescript"),
    ".jsx":  _code_parser_factory("javascript"),
    ".tsx":  _code_parser_factory("typescript"),
    ".go":   _code_parser_factory("go"),
    ".rs":   _code_parser_factory("rust"),
    ".cpp":  _code_parser_factory("cpp"),
    ".cc":   _code_parser_factory("cpp"),
    ".cxx":  _code_parser_factory("cpp"),
    ".c":    _code_parser_factory("c"),
    ".h":    _code_parser_factory("c"),
    ".hpp":  _code_parser_factory("cpp"),
    ".cs":   _code_parser_factory("csharp"),
    ".rb":   _code_parser_factory("ruby"),
    ".php":  _code_parser_factory("php"),
    ".swift": _code_parser_factory("swift"),
    ".kt":   _code_parser_factory("kotlin"),
    ".scala": _code_parser_factory("scala"),
    ".r":    _code_parser_factory("r"),
    ".sql":  _code_parser_factory("sql"),
    ".sh":   _code_parser_factory("bash"),
    ".bash": _code_parser_factory("bash"),
    ".yaml": _code_parser_factory("yaml"),
    ".yml":  _code_parser_factory("yaml"),
    ".xml":  _code_parser_factory("xml"),
    ".css":  _code_parser_factory("css"),
    ".scss": _code_parser_factory("css"),
    ".less": _code_parser_factory("css"),
    ".vue":  _code_parser_factory("vue"),
    ".svelte": _code_parser_factory("svelte"),
}

DEFAULT_SPLITTER = _sentence_parser


def _get_parser(file_path: str, chunk_size: int, chunk_overlap: int, split_separator: str):
    """根据文件扩展名查找合适的解析器工厂并调用。"""
    ext = Path(file_path).suffix.lower()
    factory = SPLITTER_MAP.get(ext, DEFAULT_SPLITTER)
    return factory(chunk_size=chunk_size, chunk_overlap=chunk_overlap, split_separator=split_separator)


class _NoOpParser:
    """不执行任何分割，每个 Document 直接作为一个 chunk（按页切片）。"""

    def get_nodes_from_documents(self, docs):
        from llama_index.core.schema import TextNode

        return [
            TextNode(
                text=doc.text,
                metadata=doc.metadata,
                relationships=doc.relationships,
            )
            for doc in docs
        ]


def _get_parser_by_category(
    file_path: str,
    chunk_strategy: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    split_separator: str = "\n\n",
):
    """根据切片策略选择合适的解析器。

    - default        → 按文件扩展名匹配（MarkdownNodeParser 等在 SPLITTER_MAP 中生效）
    - whole_file / by_page → 不分割（_NoOpParser）
    - 其他（自定义切片）→ SentenceSplitter
    """
    # 整文件切片 / 按页切片 → 不分割
    if chunk_strategy in ("whole_file", "by_page"):
        return _NoOpParser()

    # 默认切分 → 按扩展名匹配（Markdown/HTML/代码等结构化解析器在此路由）
    if chunk_strategy == "default":
        return _get_parser(file_path, chunk_size, chunk_overlap, split_separator)

    # 自定义切片 → SentenceSplitter
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator=split_separator,
    )


async def parse_document(
    file_path: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    split_separator: str | None = None,
    chunk_strategy: str = "default",
) -> list[dict]:
    """异步包装：在 thread pool 中执行同步解析逻辑，避免阻塞事件循环。"""
    chunk_size = chunk_size or config.DEFAULT_CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.DEFAULT_CHUNK_OVERLAP
    split_separator = split_separator or config.DEFAULT_SPLIT_SEPARATOR
    return await asyncio.to_thread(
        _parse_document_sync,
        file_path,
        chunk_size,
        chunk_overlap,
        split_separator,
        chunk_strategy=chunk_strategy,
    )


def _parse_document_sync(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
    split_separator: str,
    chunk_strategy: str = "default",
) -> list[dict]:
    reader = _build_reader(file_path)
    docs = reader.load_data()

    # 文件级乱码校验 —— 任何单个 doc 乱码率超过 10% 则整文件拒绝
    for doc in docs:
        gibberish_ratio = _calc_gibberish_ratio(doc.text)
        if gibberish_ratio > 0.10:
            raise ValueError(
                f"文件内容乱码率过高（{gibberish_ratio:.1%}），无法解析"
            )

    parser = _get_parser_by_category(
        file_path=file_path,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        split_separator=split_separator,
    )

    # 逐个 doc 解析，部分失败不中断（如 PDF 多页场景）
    all_nodes = []
    for doc in docs:
        try:
            nodes = parser.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)
        except Exception as e:
            logger.error(f"解析文档分片失败（file={file_path}）：{e}")
            # 跳过失败的 doc，继续处理剩余部分

    chunks = []
    for i, node in enumerate(all_nodes):
        clean_text = _sanitize_text(node.text)
        if not clean_text.strip():
            continue
        if _is_garbage(clean_text):
            continue
        page_label = node.metadata.get("page_label", 0) if node.metadata else 0
        try:
            page_num = int(page_label)
        except (ValueError, TypeError):
            page_num = 0

        chunks.append({
            "chunk_index": i,
            "content": clean_text,
            "page_num": page_num,
        })
    return chunks


async def read_file_content(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    reader = SimpleDirectoryReader(input_files=[file_path])
    docs = reader.load_data()
    return _sanitize_text("\n\n".join(doc.text for doc in docs))
