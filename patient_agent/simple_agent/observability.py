from __future__ import annotations

import json
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from patient_agent.simple_agent.config import DATA_ROOT


class RunLogger:
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or f"run-{uuid4().hex[:10]}"
        self.path = DATA_ROOT / "runs" / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, event_type: str, **data: Any) -> None:
        payload = {
            "time": time(),
            "run_id": self.run_id,
            "event_type": event_type,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def model_call(
        self,
        agent: str,
        model: str,
        latency_ms: int,
        action: str | None,
        usage: dict[str, Any] | None,
        request_id: str | None,
    ) -> None:
        self.event(
            "model_call",
            agent=agent,
            model=model,
            latency_ms=latency_ms,
            action=action,
            usage=usage,
            request_id=request_id,
        )

