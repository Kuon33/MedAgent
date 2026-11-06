"""采集智能体：只负责把信息问全，不判断科室。

设计要点：
1. `missing_slots` 由代码算（必填槽位 - 已填槽位），不依赖模型记忆；
2. 红旗由代码正则 + 否定词处理兜底，模型只是补充；
3. 一切「是否收束」的判断都有确定性兜底（轮数上限、无问题即结束），
   避免模型不返回问题导致空提问死循环。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from patient_agent.graph.config import MAX_GENERAL_TURNS, MAX_SPECIALTY_TURNS
from patient_agent.graph.core.departments import (
    DepartmentSpec,
    department_spec,
    general_spec,
    load_skill_body,
    match_red_flags,
)
from patient_agent.graph.core.llm import GraphLLM
from patient_agent.graph.core.prompts import COLLECT_PROMPT
from patient_agent.graph.core.utils import clip, non_empty, pretty_json, recent_dialogue
from patient_agent.graph.state import PatientState


def _slot_context(state: PatientState, spec: DepartmentSpec, missing: list[str]) -> str:
    department = state.get("current_department") or "未定科室"
    skill_body = load_skill_body(state.get("current_department"))
    conflict_note = _conflict_note(state)
    return f"""
当前科室上下文：{department}
采集重点：{spec.focus}

必填槽位：{"、".join(spec.required_slots)}
已填槽位：
{pretty_json(non_empty(state.get("slots")))}

缺失槽位：{"、".join(missing) if missing else "无"}
已经问过的问题（不要重复）：
{pretty_json((state.get("asked_questions") or [])[-8:])}

最近对话：
{recent_dialogue(state.get("messages") or [], limit=8)}

已有采集摘要：
{clip(state.get("collection_summary") or "无", 800)}

科室 Skill 摘要：
{clip(skill_body, 1500) or "无"}
{conflict_note}
""".strip()


def _conflict_note(state: PatientState) -> str:
    """把「文档 vs 患者口述」的冲突整理成一段，交给模型顺带确认。"""
    conflicts = state.get("slot_conflicts") or []
    if not conflicts:
        return ""
    items: list[str] = []
    for item in conflicts[:3]:
        if not isinstance(item, dict):
            continue
        slot = str(item.get("slot") or "").strip()
        if not slot:
            continue
        patient_value = str(item.get("patient_value") or "").strip()
        document_value = str(item.get("document_value") or "").strip()
        if patient_value and document_value:
            items.append(f"{slot}：病历写「{document_value}」，但患者说「{patient_value}」")
        elif document_value:
            items.append(f"{slot}：病历写「{document_value}」，请与患者确认是否仍然如此")
    if not items:
        return ""
    return "\n需要向患者确认的资料冲突：\n" + "\n".join(items)


def _red_flag_text(state: PatientState, slot_patch: dict[str, Any]) -> str:
    """用于红旗匹配的文本：最近患者原话 + 已填槽位 + 本轮新增。"""
    utterances: list[str] = []
    for message in list(state.get("messages") or [])[-10:]:
        role = getattr(message, "type", None)
        if role in ("ai", "assistant"):
            continue
        content = getattr(message, "content", "")
        if content:
            utterances.append(str(content))

    values = [str(value) for value in (state.get("slots") or {}).values() if value]
    values += [str(value) for value in (slot_patch or {}).values() if value]
    return "\n".join(utterances + values)


def collect_node(state: PatientState, llm: GraphLLM) -> PatientState:
    """一轮采集：补槽位 + 生成 1~2 个问题。

    同一个节点服务两个阶段：
    - 通用采集（current_department 为空）：按 GENERAL_SLOTS 收集基础信息；
    - 科室采集（current_department 已定）：按该科室的 required_slots 定向补齐。
    """
    department = state.get("current_department")
    spec = department_spec(department) or general_spec()
    phase = "specialty" if department else "general"

    # 轮数按科室分别计数：换科室后该科室从 0 重新算，
    # 通用采集用 "general" 这个键。
    turns_key = department or "general"
    dept_turns = state.get("dept_turns") or {}
    turns = int(dept_turns.get(turns_key, 0)) + 1
    total_turns = state.get("collection_turns", 0) + 1
    turn_cap = MAX_SPECIALTY_TURNS if department else MAX_GENERAL_TURNS

    filled_before = non_empty(state.get("slots"))
    missing = [slot for slot in spec.required_slots if slot not in filled_before]

    data = llm.ask_json(
        agent=f"collect:{department or 'general'}",
        instructions=COLLECT_PROMPT,
        user_input=_slot_context(state, spec, missing),
        temperature=0.3,
    )

    slot_patch = non_empty(data.get("slot_patch"))
    questions = [
        str(question).strip()
        for question in (data.get("questions") or [])
        if str(question).strip()
    ][:2]

    filled_after = {**filled_before, **slot_patch}
    missing_after = [slot for slot in spec.required_slots if slot not in filled_after]

    # 患者口述优先：模型这一轮写进来的槽位，来源标记为 patient。
    # 若与文档抽取值不一致，记一条冲突，下一轮顺带向患者确认。
    document_slots = state.get("document_slots") or {}
    sources = dict(state.get("slot_sources") or {})
    new_conflicts: list[dict[str, Any]] = []
    for name, value in slot_patch.items():
        doc_value = document_slots.get(name)
        if doc_value and str(doc_value).strip() == str(value).strip():
            # 与文档值一致：模型只是在复述文档，保留文档来源
            continue
        sources[name] = "patient"
        if doc_value:
            new_conflicts.append(
                {
                    "slot": name,
                    "patient_value": str(value),
                    "document_value": str(doc_value),
                    "reason": "患者口述与病历资料不一致",
                }
            )

    done = (
        str(data.get("action", "")).strip() == "collection_done"
        or not missing_after
        or turns >= turn_cap
    )
    if not questions and not done:
        # 模型既没给问题又不想结束：直接进入分诊，避免空提问死循环
        done = True

    reply = str(data.get("reply") or "").strip()
    if not reply and questions:
        reply = "；".join(questions)

    rules_hit = match_red_flags(_red_flag_text(state, slot_patch), department)
    urgent_flags = [rule.message for rule in rules_hit if rule.severity == "urgent"]
    warning_flags = [rule.message for rule in rules_hit if rule.severity != "urgent"]
    model_flags = [
        str(flag).strip()
        for flag in (data.get("red_flags") or [])
        if str(flag).strip()
    ]

    return {
        "messages": [AIMessage(content=reply)] if reply else [],
        "slots": slot_patch,
        "slot_sources": sources,
        "slot_conflicts": new_conflicts,
        "missing_slots": missing_after,
        "asked_questions": questions,
        "pending_questions": questions,
        "reply": reply,
        "collection_turns": total_turns,
        "dept_turns": {turns_key: turns},
        "collection_done": done,
        "phase": phase,
        "collection_summary": str(data.get("summary") or "").strip()
        or state.get("collection_summary", ""),
        "red_flags": urgent_flags + warning_flags + model_flags,
        "urgent_flags": urgent_flags,
        # status 供报告节点判断最终状态，采集阶段不代表流程结束
        "status": "finished" if done else "collecting",
    }


__all__ = ["collect_node"]
