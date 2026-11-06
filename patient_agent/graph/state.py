"""整个预问诊流程共享的状态。

字段分成三组，对应双 Agent 架构的职责边界：

1. 采集态（采集智能体写）：slots / missing_slots / asked_questions /
   collection_summary / current_department / dept_turns（按科室计数的字典）
2. 判断态（分诊智能体写）：triage / triage_action / triage_rounds
3. 安全态（两边都写，只增不减）：red_flags / urgent_flags

带 Annotated reducer 的字段由 LangGraph 自动合并，节点只需返回增量。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


def merge_dict(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """合并增量字典，新值覆盖旧值。"""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def merge_unique(left: list[str] | None, right: list[str] | None) -> list[str]:
    """合并列表，保持顺序且去重（红旗只增不减）。"""
    result: list[str] = []
    for item in list(left or []) + list(right or []):
        if item and item not in result:
            result.append(item)
    return result


class PatientState(TypedDict, total=False):
    """预问诊流程共享状态。"""

    # ---- 基础 ----
    messages: Annotated[list, add_messages]
    patient_id: str
    case_id: str
    profile_summary: str

    # ---- 病历文档信息抓取 ----
    media_id: str
    kind: str
    mime: str
    filename: str
    path: str
    status: str
    pages: int
    has_text_layer: bool
    text_chars: int
    doc_type: str
    sections: dict[str, str]
    fields: dict[str, list[str]]
    conflicts: list[dict[str, str]]
    unverified: list[str]
    error: str | None
    # 有新文档等待摄取。为 True 时路由会先回 ingest_document 再继续采集。
    media_pending: bool

    # 已上传媒体的清单：{media_id: {filename, kind, doc_type, pages, ...}}
    media: Annotated[dict[str, Any], merge_dict]
    # 每个槽位的来源：{"既往史": "document:出院小结.pdf"} 或 "patient"
    slot_sources: Annotated[dict[str, str], merge_dict]
    # 文档里抽到的槽位原值，保留用于与患者口述比对
    document_slots: Annotated[dict[str, Any], merge_dict]
    # 文档与患者口述冲突的槽位
    slot_conflicts: Annotated[list[dict[str, Any]], merge_unique]
    # 文档原文节选，写进报告用
    document_excerpts: list[str]

    # ---- 采集智能体的输入 ----
    collection_instructions: str

    # ---- 采集智能体的产物 ----
    slots: Annotated[dict[str, Any], merge_dict]
    missing_slots: list[str]
    asked_questions: Annotated[list[str], operator.add]
    pending_questions: list[str]
    collection_summary: str
    collection_turns: int
    collection_done: bool
    current_department: str | None
    # 按科室分别计的采集轮数：{"心内科": 2, "呼吸内科": 1, "general": 3}
    # 用 merge_dict reducer，节点只需返回 {"心内科": 3} 这样的增量。
    dept_turns: Annotated[dict[str, int], merge_dict]
    department_attempts: Annotated[list[dict], operator.add]
    switch_count: int

    # ---- 分诊智能体的产物 ----
    triage: Annotated[dict[str, Any], merge_dict]
    triage_action: str
    triage_rounds: int
    phase: str

    # ---- 安全 ----
    red_flags: Annotated[list[str], merge_unique]
    urgent_flags: Annotated[list[str], merge_unique]

    # ---- 兼容旧字段 / 收尾 ----
    facts: Annotated[dict[str, Any], merge_dict]
    action: str
    reply: str
    pending_question: str

    status: str
    report_path: str | None
    report_content: str
    appointment: dict[str, Any] | None
    booking_answer: str


__all__ = ["PatientState", "merge_dict", "merge_unique"]
