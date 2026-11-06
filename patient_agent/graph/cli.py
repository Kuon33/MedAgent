"""LangGraph 版预问诊 CLI。

用法：
    python -m patient_agent.graph.cli --patient-id patient_001

与 simple_agent 版 CLI 的区别：流程控制交给 LangGraph 图，
本模块只负责把 interrupt 抛出的问题读给患者，再把答案 resume 回去。
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from patient_agent.graph.graph import graph, make_config


NODE_LABELS = {
    "ingest_document": "病历解析",
    "collect": "问诊助手",
    "specialty_collect": "科室预问诊",
    "triage": "导诊助手",
    "report": "系统",
    "appointment": "系统",
    "booking": "系统",
    "memory": "系统",
}


def _label(node_name: str, department: str | None) -> str:
    if node_name == "specialty_collect" and department:
        return f"{department}预问诊"
    return NODE_LABELS.get(node_name, node_name)


def _print_update(node_name: str, update, department: str | None) -> None:
    """打印节点产生的、给患者看的回复。"""
    if not isinstance(update, dict):
        return
    reply = update.get("reply")
    if reply:
        print(f"\n{_label(node_name, department)}：{reply}")


def _run_until_interrupt(cfg, input_value):
    """跑图直到挂起或跑完。

    返回 ("interrupt", payload) 或 ("done", values)
    """
    for chunk in graph.stream(input_value, cfg, stream_mode="updates"):
        if "__interrupt__" in chunk:
            return "interrupt", chunk["__interrupt__"][0].value

        department = None
        try:
            department = graph.get_state(cfg).values.get("current_department")
        except Exception:
            pass

        for node_name, update in chunk.items():
            _print_update(node_name, update, department)

    return "done", graph.get_state(cfg).values


def _document_state(document: str) -> dict:
    """把本地病历文件挂到图状态上，供 ingest_document 节点解析。"""
    path = Path(document).expanduser()
    if not path.is_file():
        print(f"[warn] 文档不存在，跳过：{path}")
        return {}
    try:
        from patient_agent.graph.media.storage import save_upload

        info = save_upload(path.read_bytes(), filename=path.name, kind="document")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 文档准备失败，继续普通问诊：{exc}")
        return {}

    print(f"已载入病历文档：{info['filename']}（解析后会预填既往史等信息）")
    return {
        "media_id": info["media_id"],
        "kind": "document",
        "filename": info["filename"],
        "path": info["path"],
    }


def transcribe_audio_file(audio: str) -> str:
    """把录音转写成文字，作为患者的首句描述。

    转写失败返回空串，调用方回退到交互式输入。
    """
    path = Path(audio).expanduser()
    if not path.is_file():
        print(f"[warn] 音频不存在，跳过：{path}")
        return ""

    print("正在识别语音…（首次加载模型可能要十几秒）")
    try:
        from patient_agent.graph.media.asr import transcribe

        result = transcribe(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 语音识别失败：{exc}")
        return ""

    if not result.ok:
        print(f"[warn] {result.error or '语音识别失败'}")
        return ""

    print(f"识别结果：{result.text}")
    return result.text


def run_cli(
    patient_id: str,
    first_input: str | None = None,
    max_steps: int = 60,
    input_fn=input,
    document: str | None = None,
    audio: str | None = None,
) -> dict:
    """跑完一次完整预问诊，返回最终状态。

    document 传入病历文档路径时，会先解析并预填槽位，减少重复提问。
    audio 传入录音路径时，先转写成文字作为患者首句。
    """
    case_id = f"case-{uuid.uuid4().hex[:10]}"
    cfg = make_config(case_id)

    print(f"case_id: {case_id}")

    if first_input is None and audio:
        first_input = transcribe_audio_file(audio)

    if first_input is None:
        print("请描述你的主要不适。")
        first_input = input_fn().strip()

    # 首轮用 dict 注入初始状态；后续轮次用 Command 恢复
    pending = {
        "patient_id": patient_id,
        "case_id": case_id,
        "switch_count": 0,
        "messages": [HumanMessage(content=first_input)],
    }

    if document:
        pending.update(_document_state(document))

    for _ in range(max_steps):
        kind, payload = _run_until_interrupt(cfg, pending)

        if kind == "done":
            return payload

        question = payload.get("question", "")
        answer = input_fn(f"{question}\n> ").strip()
        pending = Command(resume=answer)

    raise RuntimeError("超过最大步数，疑似图未收敛")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph 版预问诊 CLI")
    parser.add_argument("--patient-id", required=True)
    parser.add_argument(
        "--input",
        dest="first_input",
        default=None,
        help="首句描述；不传则进入交互式输入",
    )
    parser.add_argument(
        "--document",
        default=None,
        help="病历文档路径（.pdf/.docx/.txt），会先解析并预填槽位",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="录音文件路径（.wav/.mp3/.m4a），会先转写成文字作为首句",
    )
    args = parser.parse_args()

    values = run_cli(
        args.patient_id,
        first_input=args.first_input,
        document=args.document,
        audio=args.audio,
    )

    if values.get("report_path"):
        print(f"\n医生预问诊报告已生成并保存：{values['report_path']}")
    if values.get("appointment"):
        print(f"挂号成功：{values['appointment']}")


if __name__ == "__main__":
    main()
