from pathlib import Path

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class EasyOCRReader(BaseReader):
    """使用 EasyOCR 从图片中提取文字，零系统依赖，纯 Python 实现。"""

    def load_data(self, file: str | Path, extra_info: dict | None = None) -> list[Document]:
        import easyocr
        import numpy as np
        from PIL import Image

        # OpenCV 的 imread() 在 Windows 上无法处理含中文的文件路径，
        # 改用 PIL 读取图片再转为 numpy 数组传给 EasyOCR，绕过该限制。
        model_dir = Path(__file__).resolve().parent.parent / "easyocr_models"
        reader = easyocr.Reader(
            ["ch_sim", "en"],
            gpu=False,
            model_storage_directory=str(model_dir),
        )
        image = np.array(Image.open(str(file)))
        result = reader.readtext(image, detail=0)
        text = "\n".join(result)
        return [Document(text=text, metadata=extra_info or {})]
