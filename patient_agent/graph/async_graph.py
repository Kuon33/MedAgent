"""异步版 LangGraph 图：供 FastAPI / SSE 使用。

LangGraph 的 `astream` 需要 async checkpointer，
同步的 `SqliteSaver` 会在运行时抛错，因此这里单独构建一份 async 图。

节点结构与同步版一致，见 graph.py 的说明。
"""

from __future__ import annotations

from functools import partial

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from patient_agent.graph.agents.collect import collect_node
from patient_agent.graph.agents.triage import triage_node
from patient_agent.graph.config import ASYNC_CHECKPOINT_DB
from patient_agent.graph.core.llm import GraphLLM
from patient_agent.graph.media.ingest import ingest_document_node
from patient_agent.graph.nodes import (
    appointment_node,
    ask_patient_node,
    booking_node,
    memory_node,
    report_node,
)
from patient_agent.graph.routes import (
    route_after_answer,
    route_after_appointment,
    route_after_collect,
    route_after_ingest,
    route_after_specialty,
    route_after_triage,
)
from patient_agent.graph.state import PatientState


def build_async_graph(llm: GraphLLM | None = None, checkpointer=None):
    """构建 async 版图。checkpointer 必须支持异步接口。"""
    llm = llm or GraphLLM()

    builder = StateGraph(PatientState)
    builder.add_node("ingest_document", partial(ingest_document_node, llm=llm))
    builder.add_node("collect", partial(collect_node, llm=llm))
    builder.add_node("specialty_collect", partial(collect_node, llm=llm))
    builder.add_node("triage", partial(triage_node, llm=llm))
    builder.add_node("ask_patient", ask_patient_node)
    builder.add_node("report", report_node)
    builder.add_node("appointment", appointment_node)
    builder.add_node("booking", booking_node)
    builder.add_node("memory", memory_node)

    builder.add_edge(START, "ingest_document")
    builder.add_conditional_edges(
        "ingest_document",
        route_after_ingest,
        {"collect": "collect", "specialty_collect": "specialty_collect"},
    )

    builder.add_conditional_edges(
        "collect",
        route_after_collect,
        {"ask_patient": "ask_patient", "triage": "triage", "report": "report"},
    )
    builder.add_conditional_edges(
        "specialty_collect",
        route_after_specialty,
        {"ask_patient": "ask_patient", "report": "report"},
    )
    builder.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "collect": "collect",
            "specialty_collect": "specialty_collect",
            "report": "report",
        },
    )
    builder.add_conditional_edges(
        "ask_patient",
        route_after_answer,
        {
            "collect": "collect",
            "specialty_collect": "specialty_collect",
            "ingest_document": "ingest_document",
        },
    )

    builder.add_edge("report", "appointment")
    builder.add_conditional_edges(
        "appointment",
        route_after_appointment,
        {"booking": "booking"},
    )
    builder.add_edge("booking", "memory")
    builder.add_edge("memory", END)

    return builder.compile(checkpointer=checkpointer)


async def create_async_graph(llm: GraphLLM | None = None):
    """创建并初始化 async checkpointer，返回编译好的图。

    注意：返回的 saver 必须保持存活，否则连接关闭后续调用会报
    "no active connection"。请配合 AsyncExitStack 使用，或直接用
    server.py 里的 lifespan 方式。
    """
    ASYNC_CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    saver = AsyncSqliteSaver(
        conn=await aiosqlite.connect(str(ASYNC_CHECKPOINT_DB), check_same_thread=False)
    )
    await saver.setup()
    return build_async_graph(llm=llm, checkpointer=saver)


__all__ = ["build_async_graph", "create_async_graph"]
