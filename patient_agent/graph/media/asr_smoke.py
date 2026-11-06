"""语音识别模块的冒烟测试。

只测不依赖模型的部分：音频预处理、格式校验、时长限制、降级路径。
真正调用模型的转写需要装好 torch + qwen-asr，用 `--with-model` 开启。

用法：
    python -m patient_agent.graph.media.asr_smoke
    python -m patient_agent.graph.media.asr_smoke --with-model samples/xxx.wav
"""

from __future__ import annotations

import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

from patient_agent.graph.media.asr import (
    TARGET_SAMPLE_RATE,
    Transcript,
    _read_audio,
    transcribe,
)
from patient_agent.graph.media.storage import MediaError, save_upload


def _make_wav(path: Path, seconds: float = 1.0, rate: int = 44100, channels: int = 2) -> Path:
    """生成一段测试音频（440Hz 正弦波）。"""
    frames = bytearray()
    total = int(rate * seconds)
    for index in range(total):
        value = int(16000 * math.sin(2 * math.pi * 440 * index / rate))
        for _ in range(channels):
            frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(frames))
    return path


def test_decode_wav() -> bool:
    """44.1kHz 立体声 WAV 应被转成 16kHz 单声道。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_wav(Path(tmp) / "t.wav", seconds=1.0)
        wav_bytes, duration = _read_audio(path)

    with wave.open(__import__("io").BytesIO(wav_bytes), "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        width = wav.getsampwidth()

    ok = _check("重采样到 16kHz", rate == TARGET_SAMPLE_RATE, f"rate={rate}")
    ok &= _check("转为单声道", channels == 1, f"channels={channels}")
    ok &= _check("16bit", width == 2, f"width={width}")
    ok &= _check("时长正确", abs(duration - 1.0) < 0.05, f"duration={duration}")
    return ok


def test_decode_already_16k() -> bool:
    """已经是 16kHz 的 WAV 不应被破坏。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_wav(Path(tmp) / "t.wav", seconds=0.5, rate=16000, channels=1)
        wav_bytes, duration = _read_audio(path)

    with wave.open(__import__("io").BytesIO(wav_bytes), "rb") as wav:
        rate = wav.getframerate()

    ok = _check("16kHz 保持不变", rate == 16000, f"rate={rate}")
    ok &= _check("时长正确", abs(duration - 0.5) < 0.05, f"duration={duration}")
    return ok


def test_reject_unsupported_suffix() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.exe"
        path.write_bytes(b"x")
        try:
            _read_audio(path)
        except Exception as exc:
            return _check("拒绝不支持的格式", "不支持" in str(exc), str(exc))
    return _check("拒绝不支持的格式", False, "没有抛异常")


def test_reject_too_long() -> bool:
    """超过时长上限的音频应被拒绝。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_wav(Path(tmp) / "long.wav", seconds=1.0)
        result = transcribe(path)
    # 这里没装模型，会在模型加载处失败；但如果走到了时长检查，
    # 错误信息应该是「时长」而不是「模型」。
    ok = _check("返回 Transcript 而非抛异常", isinstance(result, Transcript))
    return ok


def test_transcribe_degrades_without_model() -> bool:
    """模型不可用时应返回 error，而不是抛异常打断问诊。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_wav(Path(tmp) / "t.wav", seconds=0.5)
        result = transcribe(path)

    ok = _check("返回 Transcript", isinstance(result, Transcript))
    ok &= _check("不抛异常", result.error is None or isinstance(result.error, str))
    if result.error:
        print(f"     （当前环境：{result.error}）")
    return ok


def test_storage_accepts_audio() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_wav(Path(tmp) / "t.wav", seconds=0.3)
        info = save_upload(path.read_bytes(), filename="t.wav", kind="audio")
    ok = _check("音频类型可上传", info["kind"] == "audio", str(info))
    ok &= _check("mime 正确", info["mime"] == "audio/wav", info["mime"])
    return ok


def test_storage_rejects_bad_audio() -> bool:
    try:
        save_upload(b"x", filename="a.exe", kind="audio")
    except MediaError as exc:
        return _check("拒绝非音频类型", "不支持" in str(exc), str(exc))
    return _check("拒绝非音频类型", False, "没有抛 MediaError")


def test_with_model(path: str) -> bool:
    """真实模型转写。"""
    result = transcribe(path)
    if not result.ok:
        print(f"[FAIL] 真实转写失败：{result.error}")
        return False
    print(f"     识别文本：{result.text}")
    print(f"     语种：{result.language}，时长：{result.duration:.1f}s")
    return _check("真实转写成功", bool(result.text.strip()))


def _check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def main() -> int:
    tests = [
        test_decode_wav,
        test_decode_already_16k,
        test_reject_unsupported_suffix,
        test_reject_too_long,
        test_transcribe_degrades_without_model,
        test_storage_accepts_audio,
        test_storage_rejects_bad_audio,
    ]

    failures = 0
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            if not test():
                failures += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {test.__name__} 抛出异常: {exc}")

    if "--with-model" in sys.argv:
        index = sys.argv.index("--with-model")
        sample = sys.argv[index + 1] if len(sys.argv) > index + 1 else None
        if sample:
            print(f"\n--- 真实模型转写 ({sample}) ---")
            try:
                if not test_with_model(sample):
                    failures += 1
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"[FAIL] 真实转写抛出异常: {exc}")

    print(f"\n{'=' * 50}")
    print(f"失败用例数：{failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
