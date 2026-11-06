"""对比「带文档」与「不带文档」的采集轮次差异。

自动回答每轮问题（用固定回答），跑到流程结束，统计：
- collection_turns（采集轮次）
- 被问到的槽位（asked_questions 对应的槽位）

用法：
    python -m patient_agent.graph.media.compare
"""

from __future__ import annotations

import sys
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from patient_agent.graph.graph import graph, make_config
from patient_agent.graph.media.storage import save_upload

ANSWERS = [
    "活动后出现，休息能缓解",
    "没有胸痛，没有心慌，出汗不明显，也不气短",
    "今天感觉好一些了",
    "是",
    "没有特别不舒服的了",
]


def run(document_path: str | None) -> dict[str, Any]:
    case_id = f"case-{uuid.uuid4().hex[:10]}"
    cfg = make_config(case_id)

    pending: Any = {
        "patient_id": "patient_compare",
        "case_id": case_id,
        "switch_count": 0,
        "messages": [HumanMessage(content="我这两天有点胸闷")],
    }
    if document_path:
        from pathlib import Path

        path = Path(document_path)
        info = save_upload(path.read_bytes(), filename=path.name, kind="document")
        pending.update(
            {
                "media_id": info["media_id"],
                "kind": "document",
                "filename": info["filename"],
                "path": info["path"],
            }
        )

    asked: list[str] = []
    index = 0

    for _ in range(40):
        interrupted = False
        for chunk in graph.stream(pending, cfg, stream_mode="updates"):
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
                if payload.get("type") == "ask_patient":
                    asked.append(str(payload.get("question", "")))
                interrupted = True
                break
        if not interrupted:
            break

        answer = ANSWERS[index % len(ANSWERS)]
        index += 1
        pending = Command(resume=answer)

    values = graph.get_state(cfg).values
    return {
        "case_id": case_id,
        "collection_turns": values.get("collection_turns"),
        "asked": asked,
        "slots": values.get("slots") or {},
        "slot_sources": values.get("slot_sources") or {},
        "media": values.get("media") or {},
        "asked_questions": values.get("asked_questions") or [],
    }


def _questions_about(asked: list[str], keywords: list[str]) -> list[str]:
    """挑出「问到这些关键词」的问题——用来判断文档已有的槽位是否被重复问。"""
    return [
        question
        for question in asked
        if any(keyword in question for keyword in keywords)
    ]


HISTORY_KEYWORDS = ("既往", "病史", "慢性病", "高血压", "糖尿病", "手术", "过敏", "用药", "吃什么药", "服过什么药")


def main() -> int:
    document = sys.argv[1] if len(sys.argv) > 1 else "data/samples/出院小结.pdf"

    print("=== 不带文档 ===")
    without = run(None)
    print(f"采集轮次：{without['collection_turns']}")
    for question in without["asked"]:
        print(f"  ? {question}")
    print(f"槽位：{list(without['slots'])}")
    without_history = _questions_about(without["asked"], HISTORY_KEYWORDS)

    print("\n=== 带文档 ===")
    with_doc = run(document)
    print(f"采集轮次：{with_doc['collection_turns']}")
    for question in with_doc["asked"]:
        print(f"  ? {question}")
    print(f"槽位：{list(with_doc['slots'])}")
    print(f"预填来源：{with_doc['slot_sources']}")
    with_history = _questions_about(with_doc["asked"], HISTORY_KEYWORDS)

    prefilled = [
        name
        for name, source in with_doc["slot_sources"].items()
        if str(source).startswith("document:")
    ]

    without_turns = int(without["collection_turns"] or 0)
    with_turns = int(with_doc["collection_turns"] or 0)

    print("\n=== 对比 ===")
    print(f"采集轮次：{without_turns} -> {with_turns}")
    print(f"预填槽位（{len(prefilled)} 项）：{prefilled}")
    print(f"问到既往史/用药/过敏类问题：{len(without_history)} -> {len(with_history)}")
    for question in with_history:
        print(f"  ! 仍被问到：{question}")
    if not with_history:
        print("  ✓ 文档已有的既往史/用药/过敏信息没有重复提问")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
