from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    print(
        "[warn] 未安装 python-dotenv，.env 不会被加载。"
        "请改用系统环境变量或执行 pip install python-dotenv"
    )
except Exception as exc:  # pragma: no cover - 仅提示，不阻断
    print(f"[warn] 加载 .env 失败: {exc}")


DATA_ROOT = Path(os.getenv("PATIENT_AGENT_SIMPLE_DATA", PROJECT_ROOT / "data"))
SKILLS_ROOT = Path(os.getenv("PATIENT_AGENT_SKILLS_ROOT", PROJECT_ROOT / "skills" / "departments"))

CHECKPOINT_DB = Path(os.getenv("PATIENT_AGENT_CHECKPOINT_DB", DATA_ROOT / "checkpoints.sqlite"))
# 异步图（SSE API）单独一个库文件，避免与同步图（CLI/Gradio）产生 SQLite 锁竞争
ASYNC_CHECKPOINT_DB = Path(
    os.getenv("PATIENT_AGENT_ASYNC_CHECKPOINT_DB", DATA_ROOT / "checkpoints_api.sqlite")
)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_MODEL_URL")
DEFAULT_API_MODE = os.getenv("OPENAI_API_MODE", "auto").lower()

MAX_DEPARTMENT_SWITCHES = int(os.getenv("PATIENT_AGENT_MAX_SWITCHES", "2"))
MAX_SPECIALTY_TURNS = int(os.getenv("PATIENT_AGENT_MAX_SPECIALTY_TURNS", "4"))
# 尚未确定科室时的通用采集轮数上限
MAX_GENERAL_TURNS = int(os.getenv("PATIENT_AGENT_MAX_GENERAL_TURNS", "3"))

# ---- 媒体（病历文档 / 语音 / 图像）----
MEDIA_ROOT = Path(os.getenv("PATIENT_AGENT_MEDIA_ROOT", DATA_ROOT / "media"))
# 单份文档的页数与字符数上限，超出直接拒绝，避免 prompt 爆掉
MAX_DOC_PAGES = int(os.getenv("PATIENT_AGENT_MAX_DOC_PAGES", "30"))
MAX_DOC_CHARS = int(os.getenv("PATIENT_AGENT_MAX_DOC_CHARS", "40000"))
# 单次上传的文件大小上限（字节）
MAX_UPLOAD_BYTES = int(os.getenv("PATIENT_AGENT_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
# 判定为「扫描件」的阈值：整页平均字符数低于此值即认为没有文本层
SCAN_PAGE_CHAR_THRESHOLD = int(os.getenv("PATIENT_AGENT_SCAN_PAGE_CHAR_THRESHOLD", "50"))
# 允许上传的文档类型
ALLOWED_DOC_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}

# ---- 语音识别（Qwen3-ASR）----
# 模型权重目录（本地离线模型）
ASR_MODEL_PATH = Path(
    os.getenv("PATIENT_AGENT_ASR_MODEL_PATH", PACKAGE_ROOT / "models" / "Qwen3-ASR-1.7B")
)
# 推理设备：auto | cuda | cpu
# auto = 有 CUDA 就用 GPU，不够显存时自动 offload 到 CPU
ASR_DEVICE = os.getenv("PATIENT_AGENT_ASR_DEVICE", "auto").lower()
# 推理精度：bfloat16 | float16 | float32
ASR_DTYPE = os.getenv("PATIENT_AGENT_ASR_DTYPE", "bfloat16").lower()
# 强制指定语言；留空表示自动检测（预问诊场景建议填 Chinese）
ASR_LANGUAGE = os.getenv("PATIENT_AGENT_ASR_LANGUAGE", "Chinese") or None
# 单次转写最长音频时长（秒），超出直接拒绝，避免显存爆掉
ASR_MAX_SECONDS = int(os.getenv("PATIENT_AGENT_ASR_MAX_SECONDS", "120"))
# 允许上传的音频格式
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".mp4"}
# 转写结果的最大字符数，超出截断
ASR_MAX_CHARS = int(os.getenv("PATIENT_AGENT_ASR_MAX_CHARS", "2000"))

# 兼容旧命名
MAX_SWITCHES = MAX_DEPARTMENT_SWITCHES
MAX_SPECIALTY_TURNS_LIMIT = MAX_SPECIALTY_TURNS
