"""媒体（病历文档 / 语音 / 图像）接入层。

当前实现：
- documents.py  文档解析（PDF / DOCX / 纯文本），纯 Python，不调模型
- extract.py    模型抽取结构化槽位
- ingest.py     合并进状态（预填 slots，省掉重复提问）
- storage.py    上传文件落盘
- asr.py        语音识别（本地 Qwen3-ASR-1.7B），转写成文字回填输入框

后续的医学检查图像识别会作为新模块加在同目录下。
"""

from patient_agent.graph.media.asr import (
    ASRError,
    Transcript,
    transcribe,
    transcribe_bytes,
)
from patient_agent.graph.media.documents import (
    DocumentError,
    ParsedDoc,
    parse_document,
)
from patient_agent.graph.media.extract import Extracted, extract_slots
from patient_agent.graph.media.ingest import ingest_document_node
from patient_agent.graph.media.storage import (
    MediaError,
    media_dir,
    new_media_id,
    remove_media,
    resolve_audio,
    resolve_document,
    safe_filename,
    save_upload,
)

__all__ = [
    "ASRError",
    "DocumentError",
    "Extracted",
    "MediaError",
    "ParsedDoc",
    "Transcript",
    "extract_slots",
    "ingest_document_node",
    "media_dir",
    "new_media_id",
    "parse_document",
    "remove_media",
    "resolve_audio",
    "resolve_document",
    "safe_filename",
    "save_upload",
    "transcribe",
    "transcribe_bytes",
]
