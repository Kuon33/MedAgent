"""媒体文件落盘：保存上传的病历文档、语音、图片。

目录结构::

    data/media/{media_id}/{原始文件名}

media_id 用 uuid4 hex 前 12 位，避免文件名冲突与路径穿越。
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from uuid import uuid4

from patient_agent.graph.config import (
    ALLOWED_AUDIO_SUFFIXES,
    ALLOWED_DOC_SUFFIXES,
    MAX_UPLOAD_BYTES,
    MEDIA_ROOT,
)

MEDIA_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# 不允许出现的文件名字符（防止路径穿越 / 命令注入）
_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class MediaError(Exception):
    """媒体处理失败，message 直接给用户看。"""


def new_media_id() -> str:
    return uuid4().hex[:12]


def safe_filename(filename: str | None) -> str:
    """清洗上传文件名，只保留 basename 和安全字符。"""
    name = Path(str(filename or "")).name
    name = _UNSAFE_NAME_CHARS.sub("_", name).strip()
    if not name or name in (".", ".."):
        name = "upload"
    return name[:120]


def media_dir(media_id: str) -> Path:
    """按 media_id 取目录；media_id 必须是合法标识符。"""
    if not MEDIA_ID_RE.match(str(media_id or "")):
        raise MediaError("非法的媒体标识")
    return MEDIA_ROOT / str(media_id)


def save_upload(
    content: bytes,
    filename: str | None,
    kind: str = "document",
    suffix: str | None = None,
) -> dict:
    """保存上传内容，返回媒体描述字典。

    只做落盘与体积/类型校验，不解析内容。
    """
    if not content:
        raise MediaError("上传内容为空")

    if len(content) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise MediaError(f"文件超过 {limit_mb}MB 上限")

    safe = safe_filename(filename)
    final_suffix = (suffix or Path(safe).suffix).lower()
    if not final_suffix:
        final_suffix = ".bin"

    if kind == "document" and final_suffix not in ALLOWED_DOC_SUFFIXES:
        allowed = "、".join(sorted(ALLOWED_DOC_SUFFIXES))
        raise MediaError(f"不支持的文件类型 {final_suffix}，请上传 {allowed}")

    if kind == "audio" and final_suffix not in ALLOWED_AUDIO_SUFFIXES:
        allowed = "、".join(sorted(ALLOWED_AUDIO_SUFFIXES))
        raise MediaError(f"不支持的音频格式 {final_suffix}，请上传 {allowed}")

    media_id = new_media_id()
    target_dir = media_dir(media_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    stored_name = safe
    if Path(stored_name).suffix.lower() != final_suffix:
        stored_name = f"{Path(stored_name).stem}{final_suffix}"
    path = target_dir / stored_name
    path.write_bytes(content)

    return {
        "media_id": media_id,
        "kind": kind,
        "filename": stored_name,
        "path": str(path),
        "size": len(content),
        "mime": _guess_mime(final_suffix),
    }


def _guess_mime(suffix: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
    }.get(suffix, "application/octet-stream")


def resolve_document(media_id: str) -> tuple[Path | None, str]:
    """按 media_id 找回已上传的文档文件。

    返回 (文件路径, 文件名)；找不到返回 (None, "")。
    """
    try:
        directory = media_dir(media_id)
    except MediaError:
        return None, ""
    if not directory.exists():
        return None, ""
    candidates = [
        item
        for item in sorted(directory.iterdir())
        if item.is_file() and item.suffix.lower() in ALLOWED_DOC_SUFFIXES
    ]
    if not candidates:
        return None, ""
    return candidates[0], candidates[0].name


def resolve_audio(media_id: str) -> tuple[Path | None, str]:
    """按 media_id 找回已上传的音频文件。

    返回 (文件路径, 文件名)；找不到返回 (None, "")。
    """
    try:
        directory = media_dir(media_id)
    except MediaError:
        return None, ""
    if not directory.exists():
        return None, ""
    candidates = [
        item
        for item in sorted(directory.iterdir())
        if item.is_file() and item.suffix.lower() in ALLOWED_AUDIO_SUFFIXES
    ]
    if not candidates:
        return None, ""
    return candidates[0], candidates[0].name


def remove_media(media_id: str) -> None:
    """删除某次上传的全部文件（忽略异常）。"""
    try:
        target = media_dir(media_id)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
    except MediaError:
        pass


__all__ = [
    "MEDIA_ID_RE",
    "MediaError",
    "media_dir",
    "new_media_id",
    "remove_media",
    "resolve_audio",
    "resolve_document",
    "safe_filename",
    "save_upload",
]
