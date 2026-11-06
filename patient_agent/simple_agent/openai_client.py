from __future__ import annotations

import time
from typing import Any

from openai import NotFoundError, OpenAI

from patient_agent.simple_agent.config import (
    DEFAULT_API_KEY,
    DEFAULT_API_MODE,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
)
from patient_agent.simple_agent.observability import RunLogger
from patient_agent.simple_agent.state import LLMResponse


class OpenAIChat:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = DEFAULT_API_KEY,
        base_url: str | None = DEFAULT_BASE_URL,
        api_mode: str = DEFAULT_API_MODE,
        logger: RunLogger | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.api_mode = (api_mode or "auto").lower()
        self._resolved_api_mode = self.api_mode
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.logger = logger

    def ask(
        self,
        agent: str,
        instructions: str,
        user_input: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        started = time.time()
        try:
            response = self._create_response(instructions, user_input, tools)
        except NotFoundError:
            if self.api_mode != "auto":
                raise
            self._resolved_api_mode = "chat_completions"
            response = self._create_chat_completion(instructions, user_input)

        latency_ms = int((time.time() - started) * 1000)
        usage = getattr(response, "usage", None)
        usage_data = usage.model_dump() if hasattr(usage, "model_dump") else usage
        result = LLMResponse(
            text=self._extract_text(response),
            latency_ms=latency_ms,
            usage=usage_data,
            request_id=getattr(response, "_request_id", None),
        )
        if self.logger:
            self.logger.model_call(
                agent=agent,
                model=self.model,
                latency_ms=latency_ms,
                action=None,
                usage=usage_data,
                request_id=result.request_id,
            )
        return result

    def _create_response(
        self,
        instructions: str,
        user_input: str,
        tools: list[dict[str, Any]] | None,
    ):
        if self._resolved_api_mode == "chat_completions":
            return self._create_chat_completion(instructions, user_input)
        return self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
            tools=tools or None,
        )

    def _create_chat_completion(self, instructions: str, user_input: str):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_input},
            ],
        )

    def _extract_text(self, response) -> str:
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
