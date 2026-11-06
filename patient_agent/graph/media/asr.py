"""语音识别：把患者的录音转写成文字。

用本地 Qwen3-ASR-1.7B 模型（`patient_agent/models/Qwen3-ASR-1.7B`），
权重离线加载，不联网。

设计要点：
1. **模型懒加载 + 全局单例**。4.7GB 权重加载要十几秒，不能每次转写都重来。
   第一次调用时才加载，之后复用。
2. **模型不可用时降级**，不抛异常打断问诊。返回 error，转写文本为空，
   前端提示患者改用文字输入。
3. **音频先转 16kHz 单声道 WAV**。Qwen3-ASR 的特征提取器要 16kHz，
   浏览器录音常见的 webm/opus 必须先转码。这里用 soundfile + librosa，
   没有 ffmpeg 时 wav 直接重采样，其他格式会明确报错提示装 ffmpeg。

用法：
    from patient_agent.graph.media.asr import transcribe

    result = transcribe("path/to/recording.wav")
    if result.ok:
        print(result.text)
    else:
        print(result.error)
"""

from __future__ import annotations

import io
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patient_agent.graph.config import (
    ALLOWED_AUDIO_SUFFIXES,
    ASR_DEVICE,
    ASR_DTYPE,
    ASR_LANGUAGE,
    ASR_MAX_CHARS,
    ASR_MAX_SECONDS,
    ASR_MODEL_PATH,
)

# Qwen3-ASR 的特征提取器要求 16kHz 单声道
TARGET_SAMPLE_RATE = 16000


class ASRError(Exception):
    """语音识别失败。message 直接给用户看。"""


@dataclass
class Transcript:
    """转写结果。"""

    text: str = ""
    language: str = ""
    duration: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


# ---------------------------------------------------------------- 音频预处理
def _decode_to_wav(source: bytes, suffix: str) -> tuple[bytes, float]:
    """把任意音频字节流转成 16kHz 单声道 16bit PCM 的 WAV。

    返回 (wav_bytes, duration_seconds)。
    """
    suffix = (suffix or "").lower()

    # WAV：自己处理，不依赖 librosa（部分环境的 librosa 装不上）
    if suffix == ".wav":
        try:
            with wave.open(io.BytesIO(source), "rb") as wav:
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                rate = wav.getframerate()
                frames = wav.readframes(wav.getnframes())
        except Exception as exc:  # noqa: BLE001
            raise ASRError(f"WAV 解析失败：{exc}") from exc

        duration = len(frames) / float(rate * channels * width) if rate else 0.0
        return _resample_wav(frames, channels, width, rate), duration

    # 其他格式：需要 librosa + soundfile
    try:
        import librosa
        import soundfile as sf
    except ImportError as exc:
        raise ASRError(
            "转码该音频格式需要 librosa 与 soundfile，"
            "或改用 WAV 录音（浏览器端请设置 format=wav）"
        ) from exc

    try:
        data, rate = sf.read(io.BytesIO(source), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise ASRError(
            f"音频解码失败：{exc}。若为浏览器录音（webm/opus），"
            "请安装 ffmpeg 或改用 WAV 录音"
        ) from exc

    mono = data.mean(axis=1)
    duration = len(mono) / float(rate) if rate else 0.0

    if rate != TARGET_SAMPLE_RATE:
        # soxr_hq 需要额外的 soxr 包，装不上就退回默认重采样
        try:
            mono = librosa.resample(
                mono, orig_sr=rate, target_sr=TARGET_SAMPLE_RATE, res_type="soxr_hq"
            )
        except Exception:  # noqa: BLE001 - soxr 不可用时退回默认实现
            mono = librosa.resample(
                mono, orig_sr=rate, target_sr=TARGET_SAMPLE_RATE
            )

    buffer = io.BytesIO()
    sf.write(buffer, mono, TARGET_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), duration


def _resample_wav(frames: bytes, channels: int, width: int, rate: int) -> bytes:
    """把 WAV 帧数据转成 16kHz 单声道 16bit PCM。"""
    import array

    if width != 2:
        raise ASRError(f"暂只支持 16bit WAV（当前 {width * 8}bit），请重新录制")

    samples = array.array("h")
    samples.frombytes(frames)

    if channels > 1:
        samples = array.array("h", samples[::channels])

    if rate == TARGET_SAMPLE_RATE:
        out_samples = samples
    else:
        try:
            import numpy as np

            source = np.asarray(samples, dtype="float32")
            target_length = int(len(source) * TARGET_SAMPLE_RATE / rate)
            if target_length < 1:
                raise ASRError("音频太短")
            # 线性插值重采样，够用且不引入额外依赖
            positions = np.linspace(0, len(source) - 1, target_length)
            out_samples = np.interp(positions, np.arange(len(source)), source).astype("int16")
        except ImportError as exc:
            raise ASRError("重采样需要 numpy") from exc

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(TARGET_SAMPLE_RATE)
        wav.writeframes(out_samples.tobytes())
    return buffer.getvalue()


def _read_audio(path: str | Path) -> tuple[bytes, float]:
    """读音频文件并转成 16kHz WAV。"""
    path = Path(path)
    if not path.exists():
        raise ASRError("音频文件不存在")
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        allowed = "、".join(sorted(ALLOWED_AUDIO_SUFFIXES))
        raise ASRError(f"不支持的音频格式 {suffix or '未知'}，请上传 {allowed}")
    try:
        return _decode_to_wav(path.read_bytes(), suffix)
    except ASRError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ASRError(f"音频读取失败：{exc}") from exc


# ---------------------------------------------------------------- 模型加载
_MODEL_LOCK = threading.Lock()
_MODEL: Any = None
_MODEL_ERROR: str | None = None


def _torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name.lower(), torch.bfloat16)


def _resolve_device() -> tuple[str, bool]:
    """决定 (device, 是否允许 CPU offload)。

    显存不够时把部分层放 CPU，比纯 CPU 快，比纯 GPU 不 OOM。
    """
    import torch

    if ASR_DEVICE == "cpu":
        return "cpu", False
    if ASR_DEVICE == "cuda":
        return "cuda:0", False

    if not torch.cuda.is_available():
        return "cpu", False

    # auto：估算权重占用，显存不够就开 offload
    try:
        total = sum(p.stat().st_size for p in ASR_MODEL_PATH.glob("*.safetensors"))
        needed_gb = total / (1024**3)
    except OSError:
        needed_gb = 0.0

    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
    else:
        free_gb = 0.0

    # 留 0.8GB 余量给激活值和 KV cache
    if needed_gb > 0 and free_gb < needed_gb + 0.8:
        print(
            f"[warn] 显存不足（可用 {free_gb:.1f}GB / 需要约 {needed_gb:.1f}GB），"
            "启用 CPU offload"
        )
        return "cuda:0", True

    return "cuda:0", False


def load_model() -> Any:
    """加载 Qwen3-ASR 模型（懒加载 + 单例）。

    失败时抛 ASRError，并把原因缓存下来，避免每次调用都重试。
    """
    global _MODEL, _MODEL_ERROR

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        if _MODEL_ERROR is not None:
            raise ASRError(_MODEL_ERROR)

        if not ASR_MODEL_PATH.exists():
            _MODEL_ERROR = (
                f"未找到语音识别模型：{ASR_MODEL_PATH}。"
                "请检查 PATIENT_AGENT_ASR_MODEL_PATH 配置"
            )
            raise ASRError(_MODEL_ERROR)

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            _MODEL_ERROR = (
                f"语音识别依赖未安装（{exc}）。"
                "请执行 pip install qwen-asr librosa soundfile"
            )
            raise ASRError(_MODEL_ERROR) from exc

        device, offload = _resolve_device()
        dtype = _torch_dtype(ASR_DTYPE)

        try:
            kwargs: dict[str, Any] = {
                "dtype": dtype,
                "device_map": device,
                "max_new_tokens": 256,
            }
            if offload:
                # 部分层放 CPU，避免显存 OOM
                kwargs["max_memory"] = {0: "3GiB", "cpu": "8GiB"}
                kwargs["offload_folder"] = str(ASR_MODEL_PATH.parent / ".asr_offload")

            model = Qwen3ASRModel.from_pretrained(str(ASR_MODEL_PATH), **kwargs)
        except Exception as exc:  # noqa: BLE001
            _MODEL_ERROR = f"语音识别模型加载失败：{exc}"
            raise ASRError(_MODEL_ERROR) from exc

        _MODEL = model
        print(f"[info] 语音识别模型已加载（device={device}, dtype={dtype}, offload={offload}）")
        return _MODEL


def unload_model() -> None:
    """释放模型（测试或显存吃紧时用）。"""
    global _MODEL, _MODEL_ERROR
    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_ERROR = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - 释放失败不影响主流程
        pass


# ---------------------------------------------------------------- 转写
def transcribe(path: str | Path, language: str | None = None) -> Transcript:
    """把音频文件转写成文字。

    任何失败都返回 Transcript(error=...)，不抛异常——
    语音是可选输入通道，识别失败不该打断问诊。
    """
    # 1) 音频预处理
    try:
        wav_bytes, duration = _read_audio(path)
    except ASRError as exc:
        return Transcript(error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return Transcript(error=f"音频读取失败：{exc}")

    if duration > ASR_MAX_SECONDS:
        return Transcript(
            error=f"录音时长 {duration:.0f} 秒，超过上限 {ASR_MAX_SECONDS} 秒，请分段录制"
        )
    if duration < 0.2:
        return Transcript(error="录音太短，没有识别到有效语音")

    # 2) 加载模型并转写
    try:
        model = load_model()
    except ASRError as exc:
        return Transcript(error=str(exc))

    # qwen_asr 接受本地路径 / URL / base64 / (np.ndarray, sr)。
    # 这里传解码好的 (ndarray, sr)，避免再走一遍它内部的解码。
    try:
        import soundfile as sf

        audio_data, sample_rate = sf.read(
            io.BytesIO(wav_bytes), dtype="float32", always_2d=False
        )
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        audio_input: Any = (audio_data, sample_rate)
    except ImportError:
        # soundfile 不可用就退回传 WAV 路径（qwen_asr 内部能自己解码）
        audio_input = str(path)
    except Exception as exc:  # noqa: BLE001
        return Transcript(error=f"音频解码失败：{exc}")

    try:
        results = model.transcribe(
            audio=audio_input,
            language=language if language is not None else ASR_LANGUAGE,
        )
    except Exception as exc:  # noqa: BLE001
        return Transcript(error=f"语音识别失败：{exc}")

    if not results:
        return Transcript(error="语音识别没有返回结果")

    first = results[0]
    text = str(getattr(first, "text", "") or "").strip()
    detected = str(getattr(first, "language", "") or "")

    if not text:
        return Transcript(
            language=detected,
            duration=duration,
            error="没有识别到有效语音，请靠近麦克风重试，或改用文字输入",
        )

    truncated = len(text) > ASR_MAX_CHARS
    if truncated:
        text = text[:ASR_MAX_CHARS]

    return Transcript(
        text=text,
        language=detected,
        duration=duration,
        extra={"truncated": truncated},
    )


def transcribe_bytes(content: bytes, suffix: str = ".wav") -> Transcript:
    """直接转写音频字节（先落临时文件再走 transcribe）。"""
    import tempfile

    suffix = (suffix or ".wav").lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        temp_path = tmp.name
    try:
        return transcribe(temp_path)
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except OSError:
            pass


__all__ = [
    "ASRError",
    "TARGET_SAMPLE_RATE",
    "Transcript",
    "load_model",
    "transcribe",
    "transcribe_bytes",
    "unload_model",
]
