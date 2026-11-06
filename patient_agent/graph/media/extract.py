"""用模型把病历文本抽成结构化槽位。

只调用一次模型，输出 guided JSON。抽取结果不含诊断，也不覆盖本次主诉。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from patient_agent.graph.core.departments import DEPARTMENT_SPECS, GENERAL_SLOTS
from patient_agent.graph.core.llm import GraphLLM, LLMBadOutput
from patient_agent.graph.core.prompts import DOC_EXTRACT_INPUT, DOC_EXTRACT_PROMPT
from patient_agent.graph.media.documents import ParsedDoc


# 允许写入 slots 的槽位白名单：所有科室的必填槽位 + 通用槽位。
# 白名单之外的字段一律进 facts，防止模型自由发挥造槽位。
HISTORY_SLOTS = ("既往史", "用药史", "过敏史", "手术史", "个人史", "家族史", "婚育史")

# 只允许在「近期门诊病历」时写入的本次症状槽位。
CURRENT_VISIT_SLOTS = ("主诉", "病程")

DOC_TYPE_HINTS = {
    "discharge_summary": "出院小结",
    "outpatient_record": "门诊病历",
    "lab_report": "检验报告",
    "prescription": "处方",
    "unknown": "未知类型文档",
}


@dataclass
class Extracted:
    """抽取结果。"""

    doc_type: str = "unknown"
    doc_date: str = ""
    is_recent_visit: bool = False
    slots: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    source_quotes: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def allowed_slot_names() -> list[str]:
    """所有合法的槽位名。"""
    names: list[str] = list(GENERAL_SLOTS)
    names.extend(HISTORY_SLOTS)
    for spec in DEPARTMENT_SPECS.values():
        names.extend(spec.required_slots)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _is_recent(doc_date: str, is_recent_visit: bool, doc_type: str = "") -> bool:
    """判断文档是否可视为本次就诊资料。

    保守策略：只有「近期门诊病历」才允许写入本次主诉/病程。
    出院小结、检验报告、处方即使日期很近，也只是既往资料——
    出院小结描述的是上一次住院的情况，填进本次主诉会误导医生。
    """
    if doc_type and doc_type != "outpatient_record":
        return False
    if is_recent_visit:
        return True
    if not doc_date:
        return False

    from datetime import date, datetime

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            parsed = datetime.strptime(doc_date, fmt).date()
        except ValueError:
            continue
        return (date.today() - parsed).days <= 90
    return False


def _normalize_slots(
    raw: dict[str, Any],
    doc_date: str,
    is_recent_visit: bool,
    doc_type: str = "",
) -> dict[str, Any]:
    """过滤 + 分流模型输出的槽位。"""
    allowed = set(allowed_slot_names())
    recent = _is_recent(doc_date, is_recent_visit, doc_type)

    slots: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        name = str(key).strip()
        if not name or name not in allowed:
            continue
        text = str(value).strip()
        if not text:
            continue
        if name in CURRENT_VISIT_SLOTS and not recent:
            # 既往病历的主诉不能当本次主诉用
            continue
        slots[name] = text
    return slots


def extract_slots(
    parsed: ParsedDoc,
    llm: GraphLLM,
    slot_names: list[str] | None = None,
) -> Extracted:
    """从解析结果抽取槽位。

    抽取失败不抛异常，返回 error 非空的 Extracted，由调用方降级处理。
    """
    if parsed.needs_ocr or not parsed.text.strip():
        return Extracted(
            error=parsed.error or "文档没有可提取的文字内容",
        )

    tables = "\n".join(parsed.tables) if parsed.tables else "（无表格）"
    hint = DOC_TYPE_HINTS.get("unknown", "未知类型文档")
    names = slot_names or allowed_slot_names()

    user_input = DOC_EXTRACT_INPUT.format(
        filename=parsed.filename,
        hint=hint,
        text=parsed.text,
        tables=tables,
        slots="、".join(names),
    )

    try:
        data = llm.ask_json(
            agent="doc_extract",
            instructions=DOC_EXTRACT_PROMPT,
            user_input=user_input,
            temperature=0.0,
        )
    except LLMBadOutput as exc:
        return Extracted(error=f"文档信息抽取失败：{exc}")
    except Exception as exc:  # noqa: BLE001 - 抽取失败不阻断主流程
        return Extracted(error=f"文档信息抽取失败：{exc}")

    doc_date = str(data.get("doc_date") or "").strip()
    is_recent = bool(data.get("is_recent_visit"))
    doc_type = str(data.get("doc_type") or "unknown").strip() or "unknown"

    facts_raw = data.get("facts")
    facts: dict[str, Any] = dict(facts_raw) if isinstance(facts_raw, dict) else {}

    slots = _normalize_slots(data.get("slots"), doc_date, is_recent, doc_type)

    # 把被拦下的「本次主诉/病程」挪到 facts 的既往资料里，不丢信息
    raw_slots = data.get("slots") if isinstance(data.get("slots"), dict) else {}
    legacy: dict[str, str] = {}
    for name in CURRENT_VISIT_SLOTS:
        value = str((raw_slots or {}).get(name) or "").strip()
        if value and name not in slots:
            legacy[name] = value
    if legacy:
        past = facts.setdefault("既往资料", {})
        if isinstance(past, dict):
            past.update({f"既往{k}": v for k, v in legacy.items()})

    conflicts = [
        dict(item)
        for item in (data.get("conflicts") or [])
        if isinstance(item, dict)
    ]
    unverified = [
        str(item).strip() for item in (data.get("unverified") or []) if str(item).strip()
    ]
    quotes = {
        str(k).strip(): str(v).strip()
        for k, v in (data.get("source_quotes") or {}).items()
        if str(k).strip() and str(v).strip()
    }

    return Extracted(
        doc_type=doc_type,
        doc_date=doc_date,
        is_recent_visit=is_recent,
        slots=slots,
        facts=facts,
        conflicts=conflicts,
        unverified=unverified,
        source_quotes=quotes,
    )


__all__ = [
    "CURRENT_VISIT_SLOTS",
    "HISTORY_SLOTS",
    "Extracted",
    "allowed_slot_names",
    "extract_slots",
]
