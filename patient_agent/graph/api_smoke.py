"""SSE 接口冒烟测试：启动服务 → 走完一次预问诊 → 校验状态。

用法：
    python -m patient_agent.graph.api_smoke --case test_cardiology_001
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from pathlib import Path

import httpx
import uvicorn

from patient_agent.graph.server import app

CASES_FILE = Path(__file__).resolve().parents[2] / "tests" / "cases" / "manual_cases.json"


def load_case(case_id: str) -> dict:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    for case in cases:
        if case["id"] == case_id:
            return case
    raise KeyError(f"unknown case: {case_id}")


def run(case: dict, base_url: str) -> None:
    inputs = list(case["suggested_inputs"])
    # 每次运行用唯一 thread，避免命中上次的 checkpoint 直接返回旧状态
    thread_id = f"apitest-{case['id']}-{uuid.uuid4().hex[:8]}"
    messages: list[str] = []

    def consume(message: str | None) -> dict:
        nonlocal inputs
        resp = httpx.post(
            f"{base_url}/api/consult/stream",
            json={
                "patient_id": case["patient_id"],
                "message": message,
                "thread_id": thread_id,
            },
            timeout=180.0,
        )
        final: dict = {}
        for line in resp.text.splitlines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            kind = event.get("type")
            if kind == "node":
                print(f"[node:{event['node']}] {event.get('reply', '')[:80]}")
            elif kind == "interrupt":
                print(f"[interrupt:{event.get('interrupt_type')}] {event.get('question', '')[:80]}")
                if event.get("interrupt_type") == "confirm_appointment":
                    answer = "y"
                else:
                    answer = inputs.pop(0) if inputs else "y"
                print(f"[ans] {answer[:60]}")
                messages.append(answer)
                return consume(answer)
            elif kind == "done":
                final = event
            elif kind == "error":
                print("[error]", event.get("message"))
        return final

    final = consume(case["suggested_inputs"][0])

    print("\n================ 结果 ================")
    print("status      :", final.get("status"))
    print("department  :", final.get("department"))
    print("red_flags   :", final.get("red_flags"))
    print("report_path :", final.get("report_path"))
    print("appointment :", final.get("appointment"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="test_cardiology_001")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{args.port}"
    for _ in range(60):
        try:
            if httpx.get(f"{base_url}/health", timeout=2.0).status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
    else:
        raise RuntimeError("server did not start")

    try:
        run(load_case(args.case), base_url)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == "__main__":
    main()
