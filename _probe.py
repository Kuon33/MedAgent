"""端到端验证：报告文件名 == thread_id，且 /api/consult/report 能取到。"""

import json
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from patient_agent.graph.server import app

application = app
cfg = uvicorn.Config(application, host="127.0.0.1", port=8806, log_level="warning")
server = uvicorn.Server(cfg)
t = threading.Thread(target=server.run, daemon=True)
t.start()

base = "http://127.0.0.1:8806"
for _ in range(80):
    try:
        if httpx.get(f"{base}/health", timeout=2).status_code == 200:
            break
    except Exception:
        time.sleep(0.5)
else:
    raise SystemExit("no start")

case = json.loads(
    Path("tests/cases/manual_cases.json").read_text(encoding="utf-8")
)
case = [c for c in case if c["id"] == "test_dermatology_001"][0]
inputs = list(case["suggested_inputs"])

thread_id = "verify-report-case-001"
answers = iter(inputs)


def consume(message):
    r = httpx.post(
        f"{base}/api/consult/stream",
        json={"patient_id": case["patient_id"], "message": message, "thread_id": thread_id},
        timeout=180.0,
    )
    final = {}
    for line in r.text.splitlines():
        if not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])
        if ev.get("type") == "interrupt":
            ans = "y" if ev.get("interrupt_type") == "confirm_appointment" else next(answers)
            return consume(ans)
        if ev.get("type") == "done":
            final = ev
    return final


final = consume(case["suggested_inputs"][0])
print("thread_id        :", thread_id)
print("done.case_id     :", final.get("case_id"))
print("done.report_path :", final.get("report_path"))
print("equal            :", final.get("case_id") == thread_id)

# 用 thread_id 取报告
r = httpx.get(f"{base}/api/consult/report?case_id={thread_id}", timeout=15)
print("GET report by thread_id ->", r.status_code, len(r.text))
print("report head:", r.text[:80].replace("\n", " "))

# 文件是否真的存在
p = Path("data/reports") / f"{thread_id}.md"
print("file exists      :", p.exists())

server.should_exit = True
t.join(timeout=10)
