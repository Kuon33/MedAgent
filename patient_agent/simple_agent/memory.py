from __future__ import annotations

from patient_agent.simple_agent.config import DATA_ROOT
from patient_agent.simple_agent.openai_client import OpenAIChat
from patient_agent.simple_agent.state import CaseState, PatientMemory, utc_now
from patient_agent.simple_agent.storage import read_json, write_json


def memory_path(patient_id: str):
    return DATA_ROOT / "patients" / f"{patient_id}.json"


def load_patient_memory(patient_id: str) -> PatientMemory:
    data = read_json(memory_path(patient_id), {"patient_id": patient_id})
    return PatientMemory.model_validate(data)


def save_patient_memory(memory: PatientMemory) -> None:
    memory.updated_at = utc_now()
    write_json(memory_path(memory.patient_id), memory.model_dump())


def summarize_case(llm: OpenAIChat, case: CaseState) -> str:
    prompt = f"""
请把本次线上预问诊压缩成 120 字以内的长期记忆摘要。
只保留对下次分诊有价值的信息：主诉、关键症状、建议科室、红旗风险、重要病史。

病例：
{case.model_dump_json(indent=2)}
"""
    return llm.ask("memory", "你负责压缩患者问诊摘要。", prompt).text.strip()


def append_case_summary(llm: OpenAIChat, patient_id: str, case: CaseState) -> None:
    memory = load_patient_memory(patient_id)
    summary = summarize_case(llm, case)
    if summary:
        memory.history_summaries.append(summary)
    save_patient_memory(memory)

