"""分诊智能体：只看结构化采集结果，判断科室与紧急程度。

与采集智能体的分工：
- 采集智能体（agents/collect.py）只补信息，不判断科室；
- 分诊智能体（本文件）只做归类与风险判断，不问诊。

分诊智能体**看不到原始对话**，只看到 slots / missing_slots / red_flags，
这样它的判断不受患者表述方式干扰，输出也更稳定。

输出的 triage_action 有四种：

```text
urgent      命中急诊红旗，直接出报告
ask_more    还缺关键信息，回采集阶段补
specialize  应该换一个科室去采集（换科室计数 +1）
finalize    科室已确定且信息够了，出报告
```
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from patient_agent.graph.core.departments import (
    SUPPORTED_DEPARTMENTS,
    department_spec,
    match_red_flags,
)
from patient_agent.graph.core.llm import GraphLLM
from patient_agent.graph.core.prompts import TRIAGE_PROMPT
from patient_agent.graph.core.utils import clip, non_empty, pretty_json
from patient_agent.graph.state import PatientState


def _triage_context(state: PatientState) -> str:
    slots = non_empty(state.get("slots"))
    red_flags = list(state.get("red_flags") or [])

    return f"""
可选科室：{"、".join(SUPPORTED_DEPARTMENTS)}

当前科室：{state.get("current_department") or "未定"}

已采集信息：
{pretty_json(slots)}

仍然缺失的关键槽位：
{pretty_json(state.get("missing_slots") or [])}

已记录的急诊红旗：
{pretty_json(red_flags)}

采集摘要：
{clip(state.get("collection_summary") or "无", 600)}

已尝试过的科室：
{pretty_json(state.get("department_attempts") or [])}
""".strip()


def triage_node(state: PatientState, llm: GraphLLM) -> PatientState:
    """一次分诊：产出科室、紧急程度与缺失信息。"""
    data = llm.ask_json(
        agent="triage",
        instructions=TRIAGE_PROMPT,
        user_input=_triage_context(state),
        temperature=0.0,
    )

    raw_department = str(data.get("department") or "").strip()
    spec = department_spec(raw_department)
    department = spec.name if spec else ""
    if not department:
        # 模型给了未配置科室：回落到当前科室，避免虚构
        department = state.get("current_department") or ""

    urgency = str(data.get("urgency") or "routine").strip().lower()
    if urgency not in ("routine", "urgent"):
        urgency = "routine"

    # 红旗以代码规则为准：命中 urgent 规则时强制 urgent
    rules_hit = match_red_flags(
        " ".join(str(value) for value in (state.get("slots") or {}).values()),
        state.get("current_department"),
    )
    urgent_flags = [rule.message for rule in rules_hit if rule.severity == "urgent"]
    if urgent_flags:
        urgency = "urgent"

    current = state.get("current_department")
    missing = [str(item).strip() for item in (data.get("missing") or []) if str(item).strip()][:3]
    reply = str(data.get("reply") or "").strip()

    update: PatientState = {
        "triage": {
            "department": department,
            "urgency": urgency,
            "confidence": float(data.get("confidence") or 0.0),
            "reason": str(data.get("reason") or "").strip(),
        },
        "triage_rounds": state.get("triage_rounds", 0) + 1,
        "missing_slots": missing or state.get("missing_slots", []),
        "urgent_flags": urgent_flags,
        "reply": reply,
    }
    if reply:
        update["messages"] = [AIMessage(content=reply)]

    if urgency == "urgent":
        update["triage_action"] = "urgent"
        update["current_department"] = current
        return update

    if not department:
        # 无法归类：当作还缺信息，回到采集
        update["triage_action"] = "ask_more"
        update["current_department"] = current
        return update

    if current and department == current:
        # 已在该科室采集且信息够：收束出报告
        update["triage_action"] = "finalize"
        update["current_department"] = current
        return update

    if current:
        # 换科室：记录轨迹并计数
        attempts = list(state.get("department_attempts") or [])
        attempts.append(
            {
                "department": current,
                "reason": (state.get("triage") or {}).get("reason", ""),
                "summary": state.get("collection_summary", ""),
            }
        )
        update["department_attempts"] = attempts
        update["switch_count"] = state.get("switch_count", 0) + 1
        update["triage_action"] = "specialize"
    else:
        update["triage_action"] = "specialize"

    update["current_department"] = department
    return update


__all__ = ["triage_node"]
