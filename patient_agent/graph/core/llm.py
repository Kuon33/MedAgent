"""graph 版 LLM 客户端。

与 simple_agent.openai_client 的区别：
1. 支持 temperature（采集用高温、分诊用 0）；
2. 支持 json_mode，并在解析失败时重试一次；
3. OpenAI 客户端**延迟创建**，因此 import 本模块不需要 API key，
   也不会因为缺 key 让整个进程起不来。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import NotFoundError, OpenAI

from patient_agent.graph.config import (
    DEFAULT_API_KEY,
    DEFAULT_API_MODE,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
)


class LLMBadOutput(RuntimeError):
    """模型输出无法解析成 JSON 对象。"""


@dataclass
class LLMResult:
    text: str
    latency_ms: int = 0
    usage: dict[str, Any] | None = None
    request_id: str | None = None


def parse_json_object(text: str) -> dict[str, Any]:
    """从模型输出里抠出 JSON 对象，兼容 ```json 代码块和前后废话。"""
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?", "", clean).strip()
        clean = re.sub(r"```$", "", clean).strip()

    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))

    if not isinstance(value, dict):
        raise ValueError("model output is not a JSON object")
    return value


class GraphLLM:
    """同步 LLM 客户端，按 agent 名区分日志与模型。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = DEFAULT_API_KEY,
        base_url: str | None = DEFAULT_BASE_URL,
        api_mode: str = DEFAULT_API_MODE,
        logger: Any = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.api_mode = (api_mode or "auto").lower()
        self.logger = logger
        self._resolved_api_mode = self.api_mode
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """延迟创建客户端，避免 import 期就要求 API key。"""
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------ 调用
    def ask(
        self,
        agent: str,
        instructions: str,
        user_input: str,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResult:
        started = time.time()
        try:
            response = self._create(instructions, user_input, temperature, json_mode)
        except NotFoundError:
            if self.api_mode != "auto":
                raise
            self._resolved_api_mode = "chat_completions"
            response = self._create(instructions, user_input, temperature, json_mode)

        latency_ms = int((time.time() - started) * 1000)
        usage = getattr(response, "usage", None)
        if usage is None:
            usage_data = None
        elif hasattr(usage, "model_dump"):
            usage_data = usage.model_dump()
        else:
            usage_data = dict(usage)

        result = LLMResult(
            text=self._extract_text(response),
            latency_ms=latency_ms,
            usage=usage_data,
            request_id=getattr(response, "_request_id", None),
        )
        self._log(agent, result)
        return result

    def ask_json(
        self,
        agent: str,
        instructions: str,
        user_input: str,
        temperature: float = 0.0,
        retries: int = 1,
    ) -> dict[str, Any]:
        """要求模型输出 JSON 对象，失败时重试一次再抛错。"""
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            prompt = user_input
            if attempt:
                prompt += (
                    "\n\n上一次输出不是合法 JSON。"
                    "请只输出一个 JSON 对象，不要解释、不要使用 markdown 代码块。"
                )
            result = self.ask(
                agent,
                instructions,
                prompt,
                temperature=0.0 if attempt else temperature,
                json_mode=True,
            )
            try:
                return parse_json_object(result.text)
            except Exception as exc:  # noqa: BLE001 - 统一转成 LLMBadOutput
                last_error = exc

        raise LLMBadOutput(f"{agent} 的输出无法解析为 JSON：{last_error}") from last_error

    # ------------------------------------------------------------------ 内部
    def _create(self, instructions: str, user_input: str, temperature: float, json_mode: bool):
        if self._resolved_api_mode == "chat_completions":
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_input},
                ],
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            return self.client.chat.completions.create(**kwargs)

        return self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
            temperature=temperature,
        )

    @staticmethod
    def _extract_text(response) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text is not None:
            return output_text

        choices = getattr(response, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if content is not None:
                return content

        return str(response)

    def _log(self, agent: str, result: LLMResult) -> None:
        if not self.logger:
            return
        try:
            self.logger.model_call(
                agent=agent,
                model=self.model,
                latency_ms=result.latency_ms,
                action=None,
                usage=result.usage,
                request_id=result.request_id,
            )
        except Exception:  # pragma: no cover - 日志失败不影响主流程
            pass


__all__ = ["GraphLLM", "LLMBadOutput", "LLMResult", "parse_json_object"]
