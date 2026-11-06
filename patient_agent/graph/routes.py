"""路由函数：决定下一个节点。

独立成模块，供同步图（graph.py）和异步图（async_graph.py）共用。

新架构的流程：

```text
START -> collect -> ask_patient -> collect -> triage -> ...
```

- 采集阶段未收束 → 继续 ask_patient
- 采集收束 → triage
- triage 判定缺信息 → 回 collect
- triage 判定急症 → report
- triage 判定换科室 → specialty_collect（带该科室 skill）
- 科室采集收束 → report
- 换科室次数用尽 → report
"""

from __future__ import annotations

from patient_agent.graph.config import MAX_DEPARTMENT_SWITCHES
from patient_agent.graph.state import PatientState


def route_after_collect(state: PatientState) -> str:
    """采集节点之后：急症直接出报告，收束了就分诊，没收束就继续问。"""
    if state.get("urgent_flags"):
        return "report"
    if state.get("collection_done"):
        return "triage"
    return "ask_patient"


def route_after_triage(state: PatientState) -> str:
    """分诊节点之后：按 triage_action 决定去向。"""
    action = state.get("triage_action", "ask_more")

    if action == "urgent" or state.get("urgent_flags"):
        return "report"

    if action == "ask_more":
        return "collect"

    if action == "finalize":
        return "report"

    if action == "specialize":
        if state.get("switch_count", 0) > MAX_DEPARTMENT_SWITCHES:
            return "report"
        return "specialty_collect"

    return "collect"


def route_after_ingest(state: PatientState) -> str:
    """文档摄取之后：按当前科室回到对应的采集节点。"""
    return "specialty_collect" if state.get("current_department") else "collect"


def route_after_answer(state: PatientState) -> str:
    """患者回答后：还没定科室就回通用采集，已定科室就回科室采集。

    中途上传了病历文档时，先去摄取文档再继续采集。
    """
    if state.get("media_pending"):
        return "ingest_document"
    return "specialty_collect" if state.get("current_department") else "collect"


def route_after_specialty(state: PatientState) -> str:
    """科室采集之后：收束就出报告，否则继续追问。"""
    if state.get("urgent_flags"):
        return "report"
    if state.get("collection_done"):
        return "report"
    return "ask_patient"


def route_after_appointment(state: PatientState) -> str:
    """患者确认挂号后写入预约信息。"""
    return "booking"


__all__ = [
    "route_after_answer",
    "route_after_appointment",
    "route_after_collect",
    "route_after_ingest",
    "route_after_specialty",
    "route_after_triage",
]
