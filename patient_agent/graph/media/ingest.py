"""文档摄取节点：把上传的病历解析、抽取、合并进状态。

这个节点放在采集智能体**之前**跑。采集智能体的 missing 是拿必填槽位
和 state["slots"] 做差集现算的，所以这里把文档里的既往史/用药史/过敏史
预填进 slots 之后，采集会自动跳过这些槽位，一个问题都不用问。

合并规则（写死在代码里）：
1. 患者亲口说的优先：槽位来源是 patient 时，文档值不覆盖，只记冲突；
2. 文档内部冲突：多份文档同一槽位不一致 → 取 doc_date 最新的那份，并留痕；
3. 冲突不静默：记录到 slot_conflicts，由 collect 在下一轮顺带确认。

降级：任何一步失败都只记录 error，主流程继续，绝不崩掉整次问诊。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from patient_agent.graph.core.llm import GraphLLM
from patient_agent.graph.media.documents import DocumentError, parse_document
from patient_agent.graph.media.extract import Extracted, extract_slots
from patient_agent.graph.state import PatientState


def _source_tag(filename: str, media_id: str) -> str:
    return f"document:{media_id}:{filename}"


def _merge_slots(
    state: PatientState,
    extracted: Extracted,
    source: str,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    """合并文档槽位到已有槽位。

    返回 (合并后的 slots, slot_sources, conflicts)。
    """
    existing = dict(state.get("slots") or {})
    sources = dict(state.get("slot_sources") or {})
    conflicts: list[dict[str, Any]] = []

    for name, value in extracted.slots.items():
        current_source = sources.get(name, "")
        if current_source == "patient" and name in existing:
            # 规则 1：患者口述优先，文档不覆盖
            if str(existing.get(name, "")).strip() != str(value).strip():
                conflicts.append(
                    {
                        "slot": name,
                        "patient_value": str(existing.get(name, "")),
                        "document_value": str(value),
                        "source": source,
                        "reason": "患者口述与文档不一致",
                    }
                )
            continue

        if name in existing and sources.get(name, "").startswith("document:"):
            # 规则 2：多份文档同一槽位 → 取 doc_date 更新的那份
            conflicts.append(
                {
                    "slot": name,
                    "previous_value": str(existing.get(name, "")),
                    "document_value": str(value),
                    "source": source,
                    "reason": "多份文档信息不一致，已采用最新日期的一份",
                }
            )

        existing[name] = value
        sources[name] = source

    return existing, sources, conflicts


def ingest_document_node(state: PatientState, llm: GraphLLM) -> PatientState:
    """摄取已上传的病历文档。

    没有文档时是 no-op（只清掉 media_pending），主流程不受影响。
    """
    path = state.get("path")
    if not path:
        return {"media_pending": False}

    media_id = str(state.get("media_id") or "")
    filename = str(state.get("filename") or Path(path).name)

    # 1) 解析
    try:
        parsed = parse_document(path)
    except DocumentError as exc:
        return {
            "status": "documents_failed",
            "media_pending": False,
            "error": str(exc),
            "media": {
                media_id: {
                    "media_id": media_id,
                    "filename": filename,
                    "kind": state.get("kind") or "document",
                    "status": "failed",
                    "error": str(exc),
                }
            },
        }
    except Exception as exc:  # noqa: BLE001 - 解析失败不阻断主流程
        return {
            "status": "documents_failed",
            "media_pending": False,
            "error": f"文档解析失败：{exc}",
            "media": {
                media_id: {
                    "media_id": media_id,
                    "filename": filename,
                    "kind": state.get("kind") or "document",
                    "status": "failed",
                    "error": str(exc),
                }
            },
        }

    # 扫描件：没有文本层，提示但不阻断
    if parsed.needs_ocr:
        message = parsed.error or "文档没有可提取的文字"
        return {
            "status": "documents_need_ocr",
            "media_pending": False,
            "error": message,
            "pages": parsed.pages,
            "has_text_layer": False,
            "text_chars": parsed.text_chars,
            "media": {
                media_id: {
                    "media_id": media_id,
                    "filename": filename,
                    "kind": state.get("kind") or "document",
                    "status": "needs_ocr",
                    "pages": parsed.pages,
                    "text_chars": parsed.text_chars,
                    "has_text_layer": False,
                    "error": message,
                }
            },
        }

    # 2) 抽取
    extracted = extract_slots(parsed, llm)
    if not extracted.ok:
        return {
            "status": "documents_failed",
            "media_pending": False,
            "error": extracted.error,
            "pages": parsed.pages,
            "has_text_layer": True,
            "text_chars": parsed.text_chars,
            "media": {
                media_id: {
                    "media_id": media_id,
                    "filename": filename,
                    "kind": state.get("kind") or "document",
                    "status": "failed",
                    "pages": parsed.pages,
                    "text_chars": parsed.text_chars,
                    "has_text_layer": True,
                    "error": extracted.error,
                }
            },
        }

    # 3) 合并
    source = _source_tag(filename, media_id)
    merged_slots, slot_sources, merge_conflicts = _merge_slots(state, extracted, source)

    all_conflicts = merge_conflicts + list(extracted.conflicts)

    # 4) 组装 facts：文档里的客观检查数据单独放，不与 slots 混
    facts_patch: dict[str, Any] = {}
    if extracted.facts:
        facts_patch["既往资料"] = extracted.facts
    if extracted.doc_date:
        facts_patch.setdefault("既往资料", {})["文档日期"] = extracted.doc_date
    if extracted.doc_type:
        facts_patch.setdefault("既往资料", {})["文档类型"] = extracted.doc_type

    excerpts: list[str] = []
    for name, quote in list(extracted.source_quotes.items())[:6]:
        excerpts.append(f"[{filename}] {name}：{quote}")

    media_entry = {
        "media_id": media_id,
        "filename": filename,
        "kind": state.get("kind") or "document",
        "status": "ingested",
        "doc_type": extracted.doc_type,
        "doc_date": extracted.doc_date,
        "pages": parsed.pages,
        "text_chars": parsed.text_chars,
        "has_text_layer": True,
        "prefilled_slots": sorted(extracted.slots.keys()),
        "unverified": extracted.unverified,
    }

    return {
        "status": "documents_ingested",
        "media_pending": False,
        "doc_type": extracted.doc_type,
        "pages": parsed.pages,
        "has_text_layer": True,
        "text_chars": parsed.text_chars,
        "sections": parsed.sections,
        "slots": merged_slots,
        "slot_sources": slot_sources,
        "document_slots": extracted.slots,
        "slot_conflicts": all_conflicts,
        "facts": facts_patch,
        "fields": extracted.slots,
        "conflicts": extracted.conflicts,
        "unverified": extracted.unverified,
        "document_excerpts": excerpts,
        "media": {media_id: media_entry},
    }


__all__ = ["ingest_document_node"]
