"""图中与「问患者」无关的通用节点：追问、报告、挂号确认、写入记忆。

这些节点被同步图（CLI / Gradio）和异步图（SSE API）共用。

设计要点：
1. `ask_patient_node` 与 `appointment_node` 使用 LangGraph 的 interrupt 挂起，
   等待前端/CLI 提交回答。**这两个节点必须无副作用**——LangGraph 恢复时
   会从头重跑本节点，把写文件、扣款之类副作用放进来会执行两次。
   因此预约信息的写入被拆到后置的 `booking_node`。
2. 报告与长期记忆复用 simple_agent 的实现，保持与旧版行为一致。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage

from patient_agent.graph.core.utils import non_empty
from patient_agent.graph.state import PatientState


YES_VALUES = {"y", "yes", "是", "需要", "好", "好的", "可以", "行", "确认"}


# ---------------------------------------------------------------- 追问节点
def ask_patient_node(state: PatientState) -> PatientState:
    """挂起等待患者输入。该节点必须保持无副作用。"""
    from langgraph.types import interrupt

    questions = state.get("pending_questions") or []
    question = str(questions[0]) if questions else str(state.get("reply") or "")

    answer = interrupt(
        {
            "type": "ask_patient",
            "question": question,
            "department": state.get("current_department"),
        }
    )
    return {"messages": [HumanMessage(content=str(answer))]}


# ---------------------------------------------------------------- 报告节点
def report_node(state: PatientState) -> PatientState:
    """生成医生预问诊报告并落盘。

    报告文件名必须是 case_id（= thread_id），否则前端用 thread_id
    请求 /api/consult/report 时找不到文件。这里显式把 state 里的
    case_id 传给 CaseState，不能让它用默认的随机 id。
    """
    from patient_agent.graph.core.llm import GraphLLM
    from patient_agent.simple_agent.reporting import generate_report
    from patient_agent.simple_agent.state import CaseState

    status = state.get("status") or "finished"
    if status in ("collecting", "ready_for_triage"):
        status = "finished"
    if state.get("triage_action") == "urgent" or state.get("urgent_flags"):
        status = "urgent"

    case_id = str(state.get("case_id") or "").strip()
    if not case_id:
        # 兜底：没有 case_id 时生成一个，保证报告能落盘且可被检索
        case_id = f"case-{uuid4().hex[:10]}"

    case = CaseState(
        case_id=case_id,
        patient_id=state.get("patient_id", ""),
        messages=_messages_to_dicts(state),
        facts=_report_facts(state),
        current_department=state.get("current_department"),
        department_attempts=[],
        switch_count=state.get("switch_count", 0),
        red_flags=list(state.get("red_flags") or []),
        status=status,
    )

    llm = GraphLLM()
    report = generate_report(llm, case)

    return {
        "case_id": case_id,
        "status": status,
        "report_content": report,
        "report_path": case.report_path,
        "pending_questions": [],
    }


# ---------------------------------------------------------------- 挂号节点
def appointment_node(state: PatientState) -> PatientState:
    """挂起询问是否需要一键挂号。该节点无副作用。"""
    from langgraph.types import interrupt

    department = booking_department(state)
    answer = interrupt(
        {
            "type": "confirm_appointment",
            "question": f"是否需要一键挂号到{department}？输入 y/yes/是 确认：",
            "department": department,
        }
    )
    return {"booking_answer": str(answer)}


def booking_node(state: PatientState) -> PatientState:
    """根据患者回答写入或跳过预约。"""
    from patient_agent.simple_agent.state import utc_now

    answer = (state.get("booking_answer") or "").strip().lower()
    if answer not in YES_VALUES:
        return {"appointment": None, "reply": "已跳过一键挂号。"}

    department = booking_department(state)
    appointment = {
        "appointment_id": f"appt-{uuid4().hex[:8]}",
        "department": department,
        "status": "success",
        "created_at": utc_now(),
    }
    return {
        "appointment": appointment,
        "reply": f"挂号成功：已为您预约{department}，预约号 {appointment['appointment_id']}。",
    }


# ---------------------------------------------------------------- 记忆节点
def memory_node(state: PatientState) -> PatientState:
    """把本次问诊压缩成长期记忆摘要。"""
    from patient_agent.graph.core.llm import GraphLLM
    from patient_agent.simple_agent.memory import append_case_summary
    from patient_agent.simple_agent.state import CaseState

    patient_id = state.get("patient_id")
    if not patient_id:
        return {}

    case = CaseState(
        patient_id=patient_id,
        messages=_messages_to_dicts(state),
        facts=_report_facts(state),
        current_department=state.get("current_department"),
        department_attempts=[],
        switch_count=state.get("switch_count", 0),
        red_flags=list(state.get("red_flags") or []),
        status=state.get("status") or "finished",
    )
    if state.get("report_path"):
        case.report_path = state["report_path"]
    if state.get("appointment"):
        case.appointment = state["appointment"]

    try:
        append_case_summary(GraphLLM(), patient_id, case)
    except Exception as exc:  # pragma: no cover - 记忆失败不阻断主流程
        print(f"[warn] 写入患者记忆失败: {exc}")

    return {}


# ---------------------------------------------------------------- 工具函数
def _report_facts(state: PatientState) -> dict[str, Any]:
    """组装写进报告的事实：槽位 + 文档节选。

    文档节选带上文件名与「未经医生核实」的标注，方便医生核对原文。
    """
    facts: dict[str, Any] = {}
    facts.update(non_empty(state.get("facts")))
    facts.update(non_empty(state.get("slots")))

    excerpts = [str(item) for item in (state.get("document_excerpts") or []) if str(item).strip()]
    if excerpts:
        facts["病历资料节选（系统自动提取，未经医生核实）"] = "\n".join(excerpts)

    conflicts = [item for item in (state.get("slot_conflicts") or []) if isinstance(item, dict)]
    if conflicts:
        lines = []
        for item in conflicts[:5]:
            slot = item.get("slot", "")
            patient_value = item.get("patient_value", "")
            document_value = item.get("document_value", "")
            if patient_value and document_value:
                lines.append(f"{slot}：病历「{document_value}」/ 患者口述「{patient_value}」")
            else:
                lines.append(f"{slot}：{document_value or patient_value}")
        facts["待医生核实的资料冲突"] = "\n".join(lines)

    return facts


def booking_department(state: PatientState) -> str:
    """决定挂哪个科室。"""
    if state.get("status") == "urgent" or state.get("urgent_flags"):
        return "急诊科"
    if state.get("current_department"):
        return state["current_department"]
    return "全科医学科"


def _messages_to_dicts(state: PatientState) -> list[dict[str, str]]:
    """把 LangChain 消息转成 simple_agent 需要的 dict 列表。"""
    messages: list[dict[str, str]] = []
    for message in state.get("messages") or []:
        role = getattr(message, "type", "user")
        role = "assistant" if role in ("ai", "assistant") else "user"
        messages.append({"role": role, "content": str(getattr(message, "content", ""))})
    return messages


__all__ = [
    "appointment_node",
    "ask_patient_node",
    "booking_node",
    "booking_department",
    "memory_node",
    "report_node",
]
