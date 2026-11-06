"""小工具：把 state 压缩成模型可读的文本。"""

from __future__ import annotations

import json
from typing import Any


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def clip(text: Any, limit: int = 400) -> str:
    """截断长文本，避免 prompt 无上限膨胀。"""
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def recent_dialogue(messages: list, limit: int = 8, clip_each: int = 400) -> str:
    """取最近若干条消息拼成对话文本。"""
    lines: list[str] = []
    for message in list(messages or [])[-limit:]:
        role = getattr(message, "type", None)
        if role is None and isinstance(message, dict):
            role = message.get("role") or message.get("type")
        role = "assistant" if role in ("ai", "assistant") else "user"
        content = clip(getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else ""), clip_each)
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def non_empty(mapping: dict[str, Any] | None) -> dict[str, Any]:
    """过滤掉空值，避免空字符串覆盖已经采集到的信息。"""
    return {
        str(key): value
        for key, value in (mapping or {}).items()
        if value not in (None, "", [], {})
    }
