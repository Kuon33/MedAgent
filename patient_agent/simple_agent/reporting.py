from __future__ import annotations

from patient_agent.simple_agent.openai_client import OpenAIChat
from patient_agent.simple_agent.prompts import REPORT_PROMPT
from patient_agent.simple_agent.state import CaseState
from patient_agent.simple_agent.storage import write_report


def generate_report(llm: OpenAIChat, case: CaseState) -> str:
    prompt = f"""
病例：
{case.model_dump_json(indent=2)}
"""
    report = llm.ask("report", REPORT_PROMPT, prompt).text.strip()
    write_report(case, report)
    return report


def finish_with_report(llm: OpenAIChat, case: CaseState, status: str) -> str:
    case.status = status
    return generate_report(llm, case)

