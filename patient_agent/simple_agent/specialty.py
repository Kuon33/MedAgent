from __future__ import annotations

from patient_agent.simple_agent.config import MAX_DEPARTMENT_SWITCHES
from patient_agent.simple_agent.json_utils import parse_json_object
from patient_agent.simple_agent.openai_client import OpenAIChat
from patient_agent.simple_agent.prompts import SPECIALTY_PROMPT
from patient_agent.simple_agent.skills import load_skill, normalize_department
from patient_agent.simple_agent.state import AgentResult, CaseState, DepartmentAttempt


def run_specialty_agent(llm: OpenAIChat, case: CaseState) -> AgentResult:
    if not case.current_department:
        raise ValueError("current_department is required")

    skill = load_skill(case.current_department)
    prompt = f"""
当前科室：
{case.current_department}

科室 Skill：
{skill}

当前病例：
{case.model_dump_json(indent=2)}

请决定下一步。
"""
    response = llm.ask(f"specialty:{case.current_department}", SPECIALTY_PROMPT, prompt)
    data = parse_json_object(response.text)
    case.merge_facts(data.get("facts_patch"))
    case.merge_red_flags(data.get("red_flags"))
    result = AgentResult.from_model_data(data)
    if llm.logger:
        llm.logger.event("specialty_result", department=case.current_department, action=result.action, data=data)
    return result


def apply_handoff(case: CaseState, result: AgentResult) -> bool:
    case.switch_count += 1
    case.department_attempts.append(
        DepartmentAttempt(
            department=case.current_department or "",
            reason=result.data.get("reason", ""),
            summary=result.data.get("summary", ""),
        )
    )
    suggested = normalize_department(result.data.get("suggested_department"))
    case.current_department = suggested if suggested else None
    return case.switch_count < MAX_DEPARTMENT_SWITCHES

