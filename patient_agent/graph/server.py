"""FastAPI 服务层，为前端提供 SSE 流式接口。

启动：
    python -m patient_agent.graph.server
    或
    uvicorn patient_agent.graph.server:app --reload --port 8000

接口：
    POST /api/consult/stream   开始/继续一次预问诊（SSE 流）
    GET  /api/consult/state    查询某 thread 的当前状态
    GET  /api/consult/report   读取某 case 的报告内容
    GET  /health                健康检查
"""

from __future__ import annotations

import json
import re
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from patient_agent.graph.async_graph import build_async_graph
from patient_agent.graph.config import ASYNC_CHECKPOINT_DB, CHECKPOINT_DB, DATA_ROOT
from patient_agent.graph.graph import make_config
from patient_agent.graph.media.asr import transcribe
from patient_agent.graph.media.storage import (
    MEDIA_ID_RE,
    MediaError,
    media_dir,
    resolve_document,
    save_upload,
)


class ConsultRequest(BaseModel):
    patient_id: str
    message: str | None = Field(default=None, description="首句描述；续聊时填患者回答")
    thread_id: str | None = Field(default=None, description="会话 id；不传则新建")
    media_id: str | None = Field(default=None, description="已上传的病历文档 media_id")


class MediaUploadResponse(BaseModel):
    media_id: str
    filename: str
    size: int
    mime: str
    message: str


class AudioTranscribeResponse(BaseModel):
    media_id: str
    filename: str
    text: str
    language: str = ""
    duration: float = 0.0
    ok: bool = True
    message: str = ""


# case_id / thread_id 的合法格式：只允许字母、数字、下划线、连字符。
# 用来防止 ../../ 之类的路径穿越。
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _validate_case_id(case_id: str) -> str:
    """校验 case_id 格式，拒绝路径穿越。"""
    if not CASE_ID_PATTERN.match(case_id or ""):
        raise HTTPException(status_code=400, detail="invalid case_id")
    return case_id


def _safe_report_path(case_id: str) -> Path:
    """把 case_id 解析成报告文件的绝对路径，并确保它落在 reports 目录内。"""
    _validate_case_id(case_id)

    reports_dir = (Path(DATA_ROOT) / "reports").resolve()
    path = (reports_dir / f"{case_id}.md").resolve()

    # 双保险：即使正则被绕过，也不允许逃出 reports 目录
    if path.parent != reports_dir:
        raise HTTPException(status_code=400, detail="invalid case_id")
    return path


def _initial_media_state(media_id: str | None) -> dict[str, Any]:
    """把已上传的文档信息转成图节点的初始输入。"""
    if not media_id:
        return {}
    if not MEDIA_ID_RE.match(media_id):
        return {}
    path, filename = resolve_document(media_id)
    if path is None:
        return {}
    return {
        "media_id": media_id,
        "kind": "document",
        "filename": filename,
        "path": str(path),
    }


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _interrupt_event(payload: dict[str, Any]) -> dict[str, Any]:
    """中断事件。注意不要用 **payload 直接展开，
    payload 内部的 type 字段会覆盖外层的 type。"""
    return {
        "type": "interrupt",
        "interrupt_type": payload.get("type", "ask_patient"),
        "question": payload.get("question", ""),
        "department": payload.get("department"),
    }


def _serialize(values: dict[str, Any]) -> dict[str, Any]:
    """把 state 转成可 JSON 序列化的结构。"""
    messages = []
    for message in values.get("messages") or []:
        messages.append(
            {
                "type": getattr(message, "type", "user"),
                "content": str(getattr(message, "content", "")),
            }
        )
    return {
        "thread_id": values.get("case_id"),
        "case_id": values.get("case_id"),
        "status": values.get("status"),
        "department": values.get("current_department"),
        "phase": values.get("phase"),
        "report_path": values.get("report_path"),
        "report_content": values.get("report_content") or "",
        "appointment": values.get("appointment"),
        "facts": values.get("slots") or values.get("facts") or {},
        "missing_slots": values.get("missing_slots") or [],
        "triage": values.get("triage") or {},
        "red_flags": values.get("red_flags") or [],
        "urgent_flags": values.get("urgent_flags") or [],
        "media": values.get("media") or {},
        "slot_sources": values.get("slot_sources") or {},
        "slot_conflicts": values.get("slot_conflicts") or [],
        "messages": messages,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时创建 async checkpointer 与 async 图。"""
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncExitStack() as stack:
        saver = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(ASYNC_CHECKPOINT_DB))
        )
        await saver.setup()
        app.state.graph = build_async_graph(checkpointer=saver)
        yield


app = FastAPI(title="Patient Pre-consult Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/api/media/upload", response_model=MediaUploadResponse)
async def upload_media(file: UploadFile) -> MediaUploadResponse:
    """上传病历文档。

    只做落盘与类型/体积校验，解析与抽取在问诊开始时由图里的
    ingest_document 节点完成（需要模型）。
    """
    content = await file.read()
    try:
        info = save_upload(
            content,
            filename=file.filename,
            kind="document",
        )
    except MediaError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return MediaUploadResponse(
        media_id=info["media_id"],
        filename=info["filename"],
        size=info["size"],
        mime=info["mime"],
        message="上传成功，请开始问诊，文档信息会自动预填。",
    )


@app.get("/api/media/{media_id}")
async def get_media(media_id: str) -> dict[str, Any]:
    """查询某次上传的基本信息（不解析内容）。"""
    if not MEDIA_ID_RE.match(media_id or ""):
        raise HTTPException(status_code=400, detail="invalid media_id")
    directory = media_dir(media_id)
    if not directory.exists():
        raise HTTPException(status_code=404, detail="media not found")
    files = [item.name for item in directory.iterdir() if item.is_file()]
    return {"media_id": media_id, "files": files}


@app.post("/api/media/audio", response_model=AudioTranscribeResponse)
async def transcribe_audio(file: UploadFile) -> AudioTranscribeResponse:
    """上传录音并转写成文字。

    转写文本**不会**自动进入问诊——前端应把 text 回填到输入框，
    让患者确认或修改后再提交。这是医疗场景的必要环节：
    「胸闷」被识别成「胸焖」这类错误必须有人工确认的机会。

    模型加载较慢（首次可能十几秒），因此这里同步返回完整结果。
    """
    content = await file.read()
    try:
        info = save_upload(content, filename=file.filename, kind="audio")
    except MediaError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 识别在子线程里跑，避免阻塞事件循环
    result = await run_in_threadpool(transcribe, info["path"])

    if not result.ok:
        return AudioTranscribeResponse(
            media_id=info["media_id"],
            filename=info["filename"],
            text="",
            language=result.language,
            duration=result.duration,
            ok=False,
            message=result.error or "语音识别失败，请改用文字输入",
        )

    return AudioTranscribeResponse(
        media_id=info["media_id"],
        filename=info["filename"],
        text=result.text,
        language=result.language,
        duration=result.duration,
        ok=True,
        message="识别完成，请确认文字无误后再发送。",
    )


@app.post("/api/consult/stream")
async def consult_stream(req: ConsultRequest) -> Any:
    """开始或继续一次预问诊，以 SSE 返回事件流。

    事件类型：
        node        某个节点执行完成，携带该节点输出
        interrupt   图挂起，等待患者回答（question / department / type）
        done        流程结束，携带最终状态
        error       执行出错
    """
    graph = app.state.graph
    thread_id = req.thread_id or f"case-{uuid.uuid4().hex[:10]}"
    _validate_case_id(thread_id)
    cfg = make_config(thread_id)

    async def event_stream() -> AsyncIterator[str]:
        # 判断 thread 是否已存在：已存在说明是续聊，用 Command 恢复挂起点；
        # 不存在则用 dict 注入初始状态。
        existing = await graph.aget_state(cfg)
        is_resume = bool(existing.values)

        if req.message is not None:
            media_state = _initial_media_state(req.media_id)
            if is_resume:
                pending: Any = Command(resume=req.message)
                if media_state:
                    # 续聊时补上传文档：把文档信息写进状态，
                    # 并置 media_pending 让路由先回 ingest_document。
                    pending = Command(
                        resume=req.message,
                        update={**media_state, "media_pending": True},
                    )
            else:
                pending = {
                    "patient_id": req.patient_id,
                    "case_id": thread_id,
                    "switch_count": 0,
                    "messages": [HumanMessage(content=req.message)],
                    **media_state,
                }
        else:
            pending = Command(resume="")

        try:
            async for chunk in graph.astream(pending, cfg, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    payload = chunk["__interrupt__"][0].value
                    yield _sse(_interrupt_event(payload))
                    continue

                for node_name, update in chunk.items():
                    if isinstance(update, dict):
                        yield _sse(
                            {
                                "type": "node",
                                "node": node_name,
                                "reply": update.get("reply", ""),
                                "action": update.get("action"),
                                "department": update.get("current_department"),
                            }
                        )

            values = (await graph.aget_state(cfg)).values
            yield _sse({"type": "done", **_serialize(values)})

        except Exception as exc:  # pragma: no cover
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/consult/state")
async def consult_state(thread_id: str) -> dict[str, Any]:
    """查询某 thread 当前状态（用于前端刷新后恢复）。"""
    _validate_case_id(thread_id)
    graph = app.state.graph
    cfg = make_config(thread_id)
    snapshot = await graph.aget_state(cfg)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="thread not found")
    return _serialize(snapshot.values)


@app.get("/api/consult/report")
async def consult_report(case_id: str) -> PlainTextResponse:
    """读取某 case 的 Markdown 报告。

    case_id 必须匹配白名单格式，且解析后的路径必须仍在 reports 目录内。
    """
    path = _safe_report_path(case_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="report not found")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown")


@app.get("/json/version")
async def gradio_version() -> dict[str, Any]:
    """Gradio 版本信息。

    Gradio 5 及更早的前端会请求根路径的 /json/version，
    挂载到子路径时这个请求会 404，这里补一个兼容路由。
    """
    try:
        import gradio as gr

        return {"version": gr.__version__}
    except ImportError:
        return {"version": "unknown"}


def mount_gradio(application: FastAPI, path: str = "/") -> FastAPI:
    """把 Gradio 界面挂到现有 FastAPI 应用上，共用同一个端口。

    默认挂在根路径 "/"：Gradio 挂在子路径（如 /ui）时，页面配置里的
    api_prefix 会丢失子路径前缀，前端请求 /gradio_api/* 会 404。
    挂在根路径可以避免这个问题——业务 API 本身都在 /api/* 下，不会冲突。
    """
    try:
        import gradio as gr

        from patient_agent.graph.gradio_app import build_demo
    except ImportError:
        print("[warn] 未安装 gradio，跳过 UI 挂载。pip install gradio")
        return application

    blocks = build_demo()
    return gr.mount_gradio_app(application, blocks, path=path)


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="预问诊 Agent 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--with-gradio",
        action="store_true",
        help="同时挂载 Gradio 界面（根路径），共用同一端口",
    )
    args = parser.parse_args()

    application = app
    if args.with_gradio:
        application = mount_gradio(application)
        print(f"Gradio UI: http://127.0.0.1:{args.port}/")

    uvicorn.run(application, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
