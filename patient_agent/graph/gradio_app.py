"""Gradio 前端：把 LangGraph 预问诊图包装成一个聊天界面。

两种启动方式：

1) 独立运行（默认 7860 端口）：
       python -m patient_agent.graph.gradio_app

2) 挂到现有 FastAPI 上（与 SSE API 共用 8000 端口）：
       python -m patient_agent.graph.server --with-gradio
   然后访问 http://127.0.0.1:8000/ui

设计要点：
- 使用同步 SqliteSaver（与 CLI 一致）。Gradio 的事件在独立线程里跑，
  sqlite 连接用 check_same_thread=False，跨线程安全。
- thread_id 存在 gr.State 里，跨 submit 保持不变，用于恢复 LangGraph 的 interrupt 挂起点。
"""

from __future__ import annotations

import uuid
from pathlib import Path

import gradio as gr
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from patient_agent.graph.graph import graph, make_config
from patient_agent.graph.media.asr import transcribe
from patient_agent.graph.media.storage import MediaError, save_upload


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

YES_VALUES = {"y", "yes", "是", "需要", "好", "好的", "可以", "行", "确认"}


def _label(node: str, department: str | None) -> str:
    if node == "specialty_collect" and department:
        return f"{department}预问诊"
    return NODE_LABELS.get(node, node)


def _initial_state() -> dict:
    return {
        "thread_id": f"case-{uuid.uuid4().hex[:10]}",
        "patient_id": "patient_001",
        "started": False,
        "switch_count": 0,
        "messages": [],
        "report_md": "",
        "appointment": None,
        "media_id": None,
        "media_path": None,
        "media_filename": "",
        "media_note": "",
        "media_pending": False,
        "audio_media_id": None,
        "audio_text": "",
    }


def chat(user_input: str, state: dict):
    """处理一次用户输入，产出流式的消息与报告。

    首次调用注入初始状态；后续调用用 Command(resume=...) 恢复 interrupt 挂起点。
    """
    user_input = (user_input or "").strip()
    if not user_input:
        yield state["messages"], state["report_md"], state
        return

    thread_id = state["thread_id"]
    cfg = make_config(thread_id)

    if state["started"]:
        pending = Command(resume=user_input)
        # 中途上传病历：写进状态并置 media_pending，让路由先回 ingest_document
        if state.get("media_id") and state.get("media_path") and state.get("media_pending"):
            pending = Command(
                resume=user_input,
                update={
                    "media_id": state["media_id"],
                    "kind": "document",
                    "filename": state.get("media_filename") or Path(state["media_path"]).name,
                    "path": state["media_path"],
                    "media_pending": True,
                },
            )
    else:
        pending = {
            "patient_id": state["patient_id"],
            "case_id": thread_id,
            "switch_count": state.get("switch_count", 0),
            "messages": [HumanMessage(content=user_input)],
        }
        # 已上传病历：把它挂到初始状态，ingest_document 节点会自动预填槽位
        if state.get("media_id") and state.get("media_path"):
            pending.update(
                {
                    "media_id": state["media_id"],
                    "kind": "document",
                    "filename": state.get("media_filename") or Path(state["media_path"]).name,
                    "path": state["media_path"],
                }
            )

    messages = list(state["messages"]) + [{"role": "user", "content": user_input}]
    report_md = state["report_md"]
    appointment = state.get("appointment")

    for chunk in graph.stream(pending, cfg, stream_mode="updates"):
        if "__interrupt__" in chunk:
            payload = chunk["__interrupt__"][0].value
            kind = payload.get("type", "ask_patient")
            if kind == "confirm_appointment":
                messages.append(
                    {
                        "role": "assistant",
                        "content": payload.get("question", "是否需要一键挂号？"),
                        "metadata": {"node": "appointment"},
                    }
                )
            yield messages, report_md, {**state, "messages": messages,
                                        "started": True, "report_md": report_md}
            return

        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue

            reply = update.get("reply")
            if reply:
                department = update.get("current_department") or state.get("department")
                messages.append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "metadata": {"node": node, "label": _label(node, department)},
                    }
                )
            elif node == "ingest_document":
                messages.extend(_ingest_messages(update))

            if update.get("report_content"):
                report_md = update["report_content"]
            if update.get("appointment"):
                appointment = update["appointment"]
            if update.get("current_department"):
                state = {**state, "department": update["current_department"]}

            yield messages, report_md, {
                **state,
                "messages": messages,
                "started": True,
                "report_md": report_md,
                "appointment": appointment,
            }

    values = graph.get_state(cfg).values
    final_report = values.get("report_content") or report_md
    final_appointment = values.get("appointment") or appointment

    yield messages, final_report, {
        **state,
        "started": True,
        "messages": messages,
        "report_md": final_report,
        "appointment": final_appointment,
    }


def _ingest_messages(update: dict) -> list[dict]:
    """把文档摄取结果转成给用户看的提示消息。"""
    status = update.get("status")
    if status == "documents_ingested":
        slots = sorted((update.get("fields") or {}).keys())
        if slots:
            return [
                {
                    "role": "assistant",
                    "content": (
                        f"已读取您上传的病历文档，并自动预填 {len(slots)} 项信息"
                        f"（{'、'.join(slots)}），这些内容不会再重复问您。\n"
                        "请补充本次就诊的具体不适。"
                    ),
                    "metadata": {"node": "ingest_document", "label": "病历解析"},
                }
            ]
        return [
            {
                "role": "assistant",
                "content": "已读取您上传的文档，但没有提取到可用的病史信息，请直接描述本次不适。",
                "metadata": {"node": "ingest_document", "label": "病历解析"},
            }
        ]
    if status == "documents_need_ocr":
        return [
            {
                "role": "assistant",
                "content": (
                    "这份文档看起来是扫描件，识别质量有限，建议您用文字补充既往史、用药和过敏情况。"
                ),
                "metadata": {"node": "ingest_document", "label": "病历解析"},
            }
        ]
    if status == "documents_failed":
        return [
            {
                "role": "assistant",
                "content": (
                    f"文档没能解析成功（{update.get('error') or '未知原因'}），"
                    "请手动补充相关信息。"
                ),
                "metadata": {"node": "ingest_document", "label": "病历解析"},
            }
        ]
    return []


def upload_document(file_obj, state: dict):
    """保存上传的病历文档，返回提示文案与更新后的状态。"""
    if file_obj is None:
        return "", state

    path = Path(str(file_obj))
    if not path.exists():
        return "上传失败：文件不存在。", state

    try:
        info = save_upload(path.read_bytes(), filename=path.name, kind="document")
    except MediaError as exc:
        return f"上传失败：{exc}", state
    except Exception as exc:  # noqa: BLE001
        return f"上传失败：{exc}", state

    note = (
        f"已上传 `{info['filename']}`，开始问诊时会自动解析并预填既往史等信息。"
        "请在下方描述本次就诊的主要不适。"
    )
    new_state = {
        **state,
        "media_id": info["media_id"],
        "media_path": info["path"],
        "media_filename": info["filename"],
        "media_note": note,
        # 已开始的会话走「中途补传」路径，需要路由先回 ingest_document
        "media_pending": bool(state.get("started")),
    }
    return note, new_state


def transcribe_audio(audio_obj, state: dict):
    """把录音转写成文字，回填到输入框让患者确认。

    转写结果**不直接发送**——患者可以先核对修改。
    医疗场景里「胸闷」识别成「胸焖」这类错误必须有纠正机会。
    """
    if audio_obj is None:
        return "", "", state

    path = Path(str(audio_obj))
    if not path.exists():
        return "", "录音文件不存在，请重新录音。", state

    try:
        info = save_upload(path.read_bytes(), filename=path.name, kind="audio")
    except MediaError as exc:
        return "", f"上传失败：{exc}", state
    except Exception as exc:  # noqa: BLE001
        return "", f"上传失败：{exc}", state

    result = transcribe(info["path"])

    if not result.ok:
        return "", result.error or "语音识别失败，请改用文字输入。", state

    note = (
        f"识别完成（{result.duration:.0f} 秒"
        + (f"，识别语种：{result.language}" if result.language else "")
        + "）。**请核对文字无误后再发送**，可以直接在下方修改。"
    )
    new_state = {
        **state,
        "audio_media_id": info["media_id"],
        "audio_text": result.text,
    }
    return result.text, note, new_state


def reset():
    """开始新的问诊。"""
    return [], "", _initial_state()


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="线上预问诊 Agent") as demo:
        st = gr.State(_initial_state())

        gr.Markdown(
            "# 线上预问诊 Agent\n"
            "可以说话也可以打字：录音后系统会转成文字，**请核对无误再发送**。\n"
            "也可先上传既往病历文档（PDF / DOCX），系统会自动提取既往史、用药史、过敏史，"
            "减少重复提问。"
        )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Accordion("上传病历文档（可选）", open=False):
                    doc_file = gr.File(
                        label="病历 / 出院小结 / 门诊病历",
                        file_types=[".pdf", ".docx", ".txt", ".md"],
                        type="filepath",
                    )
                    upload_btn = gr.Button("上传并解析", size="sm")
                    upload_note = gr.Markdown(label="上传结果")

                chatbot = gr.Chatbot(
                    height=520,
                    label="问诊对话",
                )
                with gr.Row():
                    txt = gr.Textbox(
                        placeholder="描述你的不适…（或用下方麦克风录音）",
                        show_label=False,
                        container=False,
                        scale=5,
                    )
                    send_btn = gr.Button("发送", scale=1, variant="primary")
                with gr.Row():
                    mic = gr.Audio(
                        label="语音输入",
                        sources=["microphone"],
                        type="filepath",
                        format="wav",
                        scale=3,
                    )
                    asr_btn = gr.Button("识别语音", scale=1)
                asr_note = gr.Markdown(label="识别结果")
                reset_btn = gr.Button("重新问诊", size="sm")

            with gr.Column(scale=2):
                report_md_box = gr.Markdown(label="医生预问诊报告")
                appointment_box = gr.Markdown(label="挂号信息")

        def submit(text, state):
            for messages, report, new_state in chat(text, state):
                appointment = new_state.get("appointment")
                appt_md = ""
                if appointment:
                    appt_md = (
                        f"**挂号成功**\n\n"
                        f"- 科室：{appointment.get('department')}\n"
                        f"- 预约号：`{appointment.get('appointment_id')}`"
                    )
                yield messages, report, appt_md, new_state

        def do_upload(file_obj, state):
            note, new_state = upload_document(file_obj, state)
            return note, new_state

        def do_transcribe(audio_obj, state):
            text, note, new_state = transcribe_audio(audio_obj, state)
            return text, note, new_state

        def do_reset():
            messages, report, state = reset()
            yield messages, report, "", state

        upload_btn.click(
            do_upload, inputs=[doc_file, st], outputs=[upload_note, st]
        )

        asr_btn.click(
            do_transcribe, inputs=[mic, st], outputs=[txt, asr_note, st]
        )

        for trigger in (txt.submit, send_btn.click):
            trigger(submit, inputs=[txt, st], outputs=[chatbot, report_md_box, appointment_box, st])
        txt.submit(lambda: "", None, txt)

        reset_btn.click(
            do_reset, outputs=[chatbot, report_md_box, appointment_box, st]
        )

    return demo


demo = build_demo()


def main() -> None:
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
