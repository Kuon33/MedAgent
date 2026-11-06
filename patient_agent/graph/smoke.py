"""LangGraph 版预问诊的端到端冒烟测试（不依赖前端）。

用法：
    python -m patient_agent.graph.smoke --case test_cardiology_001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patient_agent.graph.cli import run_cli

CASES_FILE = Path(__file__).resolve().parents[2] / "tests" / "cases" / "manual_cases.json"


def load_case(case_id: str) -> dict:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    for case in cases:
        if case["id"] == case_id:
            return case
    raise KeyError(f"unknown case: {case_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    args = parser.parse_args()

    case = load_case(args.case)
    inputs = list(case["suggested_inputs"])

    def fake_input(prompt: str = "") -> str:
        print(f"[ASK] {prompt}")
        if not inputs:
            return "y"
        answer = inputs.pop(0)
        print(f"[ANS] {answer}")
        return answer

    values = run_cli(case["patient_id"], first_input=None, input_fn=fake_input)

    print("\n================ 结果 ================")
    print("status          :", values.get("status"))
    print("department      :", values.get("current_department"))
    print("phase           :", values.get("phase"))
    print("switch_count    :", values.get("switch_count"))
    print("dept_turns      :", values.get("dept_turns"))
    print("collection_turns:", values.get("collection_turns"))
    print("red_flags       :", values.get("red_flags"))
    print("urgent_flags    :", values.get("urgent_flags"))
    print("report_path     :", values.get("report_path"))
    print("appointment     :", values.get("appointment"))
    print("slots           :", json.dumps(values.get("slots"), ensure_ascii=False, indent=2))
    print("missing_slots   :", values.get("missing_slots"))


if __name__ == "__main__":
    main()
