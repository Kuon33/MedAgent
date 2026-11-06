from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from patient_agent.simple_agent.config import DATA_ROOT
from patient_agent.simple_agent.state import CaseState


def ensure_data_dirs() -> None:
    for name in ("patients", "cases", "runs", "reports"):
        (DATA_ROOT / name).mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_case(case: CaseState) -> Path:
    ensure_data_dirs()
    path = DATA_ROOT / "cases" / f"{case.case_id}.json"
    write_json(path, case.model_dump())
    return path


def load_case(case_id: str) -> CaseState:
    path = DATA_ROOT / "cases" / f"{case_id}.json"
    return CaseState.model_validate(read_json(path, {}))


def write_report(case: CaseState, content: str) -> Path:
    ensure_data_dirs()
    path = DATA_ROOT / "reports" / f"{case.case_id}.md"
    path.write_text(content, encoding="utf-8")
    case.report_path = str(path)
    return path

