from pathlib import Path

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class EasyOCRReader(BaseReader):
    """使用 EasyOCR 从图片中提取文字，零系统依赖，纯 Python 实现。"""

    def load_data(self, file: str | Path, extra_info: dict | None = None) -> list[Document]:
        import easyocr

        reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
        result = reader.readtext(str(file), detail=0)
        text = "\n".join(result)
        return [Document(text=text, metadata=extra_info or {})]
