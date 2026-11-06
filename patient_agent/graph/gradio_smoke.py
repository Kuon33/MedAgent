"""Gradio 界面逻辑的端到端冒烟测试（不启动浏览器）。

用法：
    python -m patient_agent.graph.gradio_smoke --case test_cardiology_001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patient_agent.graph.gradio_app import _initial_state, chat


CASES_FILE = Path(__file__).resolve().parents[2] / "tests" / "cases" / "manual_cases.json"
CONFIRM_MARKERS = ("一键挂号", "确认：")


def load_case(case_id: str) -> dict:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    for case in cases:
        if case["id"] == case_id:
            return case
    raise KeyError(f"unknown case: {case_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="test_cardiology_001")
    parser.add_argument("--max-rounds", type=int, default=12)
    args = parser.parse_args()

    case = load_case(args.case)
    inputs = list(case["suggested_inputs"])

    state = _initial_state()
    state["patient_id"] = case["patient_id"]

    for _ in range(args.max_rounds):
        if state.get("appointment"):
            break

        # 优先用测试用例给的输入；若上一条在问是否挂号，用 y 应答
        if inputs:
            text = inputs.pop(0)
        else:
            last = (state.get("messages") or [{}])[-1]
            content = str(last.get("content", ""))
            if any(marker in content for marker in CONFIRM_MARKERS):
                text = "y"
            else:
                break

        print(f"\n[用户] {text}")
        last_state = None
        for messages, report, new_state in chat(text, state):
            last_state = new_state
        if last_state is None:
            break
        state = last_state

        for message in (state.get("messages") or [])[-2:]:
            if message["role"] == "assistant":
                print(f"[助手] {message['content'][:100]}")

    print("\n================ 结果 ================")
    print("报告长度   :", len(state.get("report_md") or ""))
    print("预约信息   :", state.get("appointment"))
    print("消息条数   :", len(state.get("messages") or []))


if __name__ == "__main__":
    main()
