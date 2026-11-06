from __future__ import annotations

from patient_agent.simple_agent.json_utils import parse_json_object, to_pretty_json
from patient_agent.simple_agent.openai_client import OpenAIChat
from patient_agent.simple_agent.prompts import TRIAGE_PROMPT
from patient_agent.simple_agent.skills import load_all_skill_summaries, simple_skill_search
from patient_agent.simple_agent.state import AgentResult, CaseState, PatientMemory


def run_triage_agent(llm: OpenAIChat, case: CaseState, memory: PatientMemory) -> AgentResult:
    text_blob = " ".join(message["content"] for message in case.messages)
    rag_hits = simple_skill_search(text_blob)
    prompt = f"""
患者长期摘要：
{memory.profile_summary or "无"}

历史问诊摘要：
{to_pretty_json(memory.history_summaries[-5:])}

当前病例：
{case.model_dump_json(indent=2)}

本地科室关键词命中：
{to_pretty_json(rag_hits)}

可用科室 Skill 摘要：
{load_all_skill_summaries()}

请判断下一步。
"""
    response = llm.ask("triage", TRIAGE_PROMPT, prompt)
    data = parse_json_object(response.text)
    case.merge_facts(data.get("facts_patch"))
    case.merge_red_flags(data.get("red_flags"))
    result = AgentResult.from_model_data(data)
    if llm.logger:
        llm.logger.event("triage_result", action=result.action, data=data)
    return result

