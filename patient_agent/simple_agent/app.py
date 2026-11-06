from __future__ import annotations

import argparse
from uuid import uuid4

from patient_agent.simple_agent.config import MAX_SPECIALTY_TURNS
from patient_agent.simple_agent.memory import append_case_summary, load_patient_memory
from patient_agent.simple_agent.observability import RunLogger
from patient_agent.simple_agent.openai_client import OpenAIChat
from patient_agent.simple_agent.reporting import finish_with_report
from patient_agent.simple_agent.skills import normalize_department
from patient_agent.simple_agent.specialty import apply_handoff, run_specialty_agent
from patient_agent.simple_agent.state import CaseState, utc_now
from patient_agent.simple_agent.storage import ensure_data_dirs, save_case
from patient_agent.simple_agent.triage import run_triage_agent


YES_VALUES = {"y", "yes", "是", "需要", "好", "好的", "可以", "行", "确认"}


def ask_patient(prefix: str = "患者：") -> str:
    return input(prefix).strip()


def booking_department(case: CaseState) -> str:
    if case.status == "urgent" or case.red_flags:
        return "急诊科"
    if case.current_department:
        return case.current_department
    return "全科医学科"


def offer_appointment(case: CaseState) -> None:
    department = booking_department(case)
    answer = ask_patient(f"是否需要一键挂号到{department}？输入 y/yes/是 确认：")
    if answer.strip().lower() in YES_VALUES:
        appointment_id = f"appt-{uuid4().hex[:8]}"
        case.appointment = {
            "appointment_id": appointment_id,
            "department": department,
            "status": "success",
            "created_at": utc_now(),
        }
        print(f"挂号成功：已为您预约{department}，预约号 {appointment_id}。")
    else:
        print("已跳过一键挂号。")


def finish_case(llm: OpenAIChat, case: CaseState, status: str, message: str = "") -> None:
    if message:
        print("\n" + message)
        case.add_message("assistant", message)
    finish_with_report(llm, case, status)
    print(f"医生预问诊报告已生成并保存：{case.report_path}")
    offer_appointment(case)
    save_case(case)


def run_specialty_loop(llm: OpenAIChat, case: CaseState) -> bool:
    specialty_turns = 0

    while True:
        result = run_specialty_agent(llm, case)

        if result.action == "ask_more":
            print(f"{case.current_department}预问诊：{result.reply}")
            case.add_message("assistant", result.reply)
            case.add_message("user", ask_patient())
            specialty_turns += 1
            save_case(case)
            if specialty_turns >= MAX_SPECIALTY_TURNS:
                finish_case(
                    llm,
                    case,
                    "finished_by_turn_limit",
                    f"{case.current_department}预问诊已收集到主要信息，将停止继续追问。",
                )
                return True
            continue

        if result.action == "handoff":
            can_continue = apply_handoff(case, result)
            case.add_message("system", f"科室回退：{result.data.get('summary', '')}")
            save_case(case)
            if not can_continue:
                finish_case(llm, case, "need_offline_general_assessment", "已达到科室切换上限，建议线下综合评估。")
                return True
            return False

        if result.action == "urgent":
            finish_case(llm, case, "urgent", result.reply)
            return True

        if result.action == "specialty_done":
            finish_case(llm, case, "finished", result.reply)
            return True

        finish_case(llm, case, "finished", result.reply)
        return True


def run_case(patient_id: str) -> CaseState:
    ensure_data_dirs()
    logger = RunLogger()
    llm = OpenAIChat(logger=logger)
    memory = load_patient_memory(patient_id)
    case = CaseState(patient_id=patient_id)

    logger.event("case_started", patient_id=patient_id, case_id=case.case_id)
    print(f"case_id: {case.case_id}")
    print("请描述你的主要不适。")

    case.add_message("user", ask_patient())

    while True:
        if not case.current_department:
            result = run_triage_agent(llm, case, memory)

            if result.action == "ask_more":
                print(f"导诊助手：{result.reply}")
                case.add_message("assistant", result.reply)
                case.add_message("user", ask_patient())
                save_case(case)
                continue

            if result.action == "urgent":
                finish_case(llm, case, "urgent", result.reply)
                break

            department = normalize_department(result.data.get("department"))
            if not department:
                print("导诊助手：目前还不能判断科室，请再补充症状。")
                case.add_message("user", ask_patient())
                continue

            case.current_department = department
            case.status = "specialty"
            print(f"导诊助手：初步建议进入 {department} 预问诊。{result.data.get('reason', '')}")
            save_case(case)

        finished = run_specialty_loop(llm, case)
        if finished:
            break

    save_case(case)
    append_case_summary(llm, patient_id, case)
    logger.event("case_finished", status=case.status, report_path=case.report_path, appointment=case.appointment)
    return case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()
    run_case(args.patient_id)


if __name__ == "__main__":
    main()

