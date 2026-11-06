from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


Action = Literal[
    "ask_more",
    "triage_done",
    "specialty_done",
    "handoff",
    "urgent",
    "finish",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PatientMemory(BaseModel):
    patient_id: str
    profile_summary: str = ""
    history_summaries: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now)


class DepartmentAttempt(BaseModel):
    department: str
    reason: str = ""
    summary: str = ""
    created_at: str = Field(default_factory=utc_now)


class CaseState(BaseModel):
    case_id: str = Field(default_factory=lambda: f"case-{uuid4().hex[:10]}")
    patient_id: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
    current_department: str | None = None
    department_attempts: list[DepartmentAttempt] = Field(default_factory=list)
    switch_count: int = 0
    red_flags: list[str] = Field(default_factory=list)
    status: str = "triage"
    report_path: str | None = None
    appointment: dict[str, str] | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    def add_message(self, role: str, content: str) -> None:
        if content.strip():
            self.messages.append({"role": role, "content": content.strip()})
            self.updated_at = utc_now()

    def merge_facts(self, patch: dict[str, Any] | None) -> None:
        if not patch:
            return
        for key, value in patch.items():
            if value not in (None, "", [], {}):
                self.facts[key] = value
        self.updated_at = utc_now()

    def merge_red_flags(self, flags: list[str] | None) -> None:
        for flag in flags or []:
            if flag and flag not in self.red_flags:
                self.red_flags.append(flag)
        self.updated_at = utc_now()


class AgentResult(BaseModel):
    action: Action
    reply: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_model_data(cls, data: dict[str, Any], default_action: Action = "ask_more"):
        action = data.get("action") or default_action
        return cls(action=action, reply=data.get("reply", ""), data=data)


class LLMResponse(BaseModel):
    text: str
    latency_ms: int
    usage: dict[str, Any] | None = None
    request_id: str | None = None


