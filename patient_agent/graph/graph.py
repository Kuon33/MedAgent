from __future__ import annotations

from contextlib import ExitStack
from functools import partial

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from patient_agent.graph.agents.collect import collect_node
from patient_agent.graph.agents.triage import triage_node
from patient_agent.graph.config import CHECKPOINT_DB
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


def build_graph(llm: GraphLLM | None = None, checkpointer=None):
    """构建同步版图，供 CLI 与 Gradio 使用。

    节点结构（双 Agent 架构）：

        START -> collect -> ask_patient -> collect -> triage -> specialty_collect
                       -> report -> appointment -> booking -> memory -> END
    """
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


# 模块级 checkpointer：SqliteSaver.from_conn_string 是上下文管理器，
# 用 ExitStack 保持连接在整个进程生命周期内有效。
_stack = ExitStack()
CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
_checkpointer = _stack.enter_context(SqliteSaver.from_conn_string(str(CHECKPOINT_DB)))

# 模块级 graph，供 CLI 与 API 复用
graph = build_graph(checkpointer=_checkpointer)


def make_config(thread_id: str) -> dict:
    """每次问诊用独立 thread_id，避免患者历史串到新会话。"""
    return {"configurable": {"thread_id": thread_id}}


__all__ = [
    "build_graph",
    "graph",
    "make_config",
]
