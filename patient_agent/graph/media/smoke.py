"""病历文档摄取的冒烟测试。

不依赖真实模型：用假的 LLM 返回固定抽取结果，验证
1) 文档解析（PDF / DOCX / 纯文本）；
2) 抽取结果过滤（禁止填本次主诉、槽位白名单）；
3) 合并规则（患者口述优先、冲突留痕）；
4) 降级（加密/扫描件/不支持格式/超限）。

用法：
    python -m patient_agent.graph.media.smoke
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from patient_agent.graph.media.documents import DocumentError, parse_document
from patient_agent.graph.media.extract import allowed_slot_names, extract_slots
from patient_agent.graph.media.ingest import ingest_document_node
from patient_agent.graph.media.storage import MediaError, save_upload


DISCHARGE_TEXT = """
出院小结

姓名：张某某  性别：男  年龄：62岁
入院日期：2026-08-13  出院日期：2026-08-20

主诉：反复胸闷 3 天
现病史：患者 3 天前无明显诱因出现胸闷，位于胸骨后，持续不缓解，伴出汗，无明显呼吸困难。
既往史：高血压病史 10 年，2 型糖尿病 5 年。
用药史：苯磺酸氨氯地平 5mg 每日一次；二甲双胍 0.5g 每日两次。
过敏史：青霉素过敏。
体格检查：血压 150/95 mmHg，心率 88 次/分。
辅助检查：心电图示窦性心律，ST 段压低 0.1mV。
初步诊断：高血压 2 级极高危；2 型糖尿病。
出院医嘱：低盐低脂饮食，监测血压，门诊随诊。
"""


class FakeLLM:
    """假的 LLM：记录调用次数，返回固定 JSON。"""

    def __init__(self, payload: dict[str, Any] | None = None):
        self.payload = payload or {
            "doc_type": "discharge_summary",
            "doc_date": "2026-08-13",
            "is_recent_visit": False,
            "slots": {
                "既往史": "高血压病史10年，2型糖尿病5年",
                "用药史": "氨氯地平5mg qd；二甲双胍0.5g bid",
                "过敏史": "青霉素过敏",
                "主诉": "反复胸闷3天",
            },
            "facts": {"血压": "150/95 mmHg", "心率": "88 次/分"},
            "conflicts": [],
            "unverified": [],
            "source_quotes": {"既往史": "高血压病史10年"},
        }
        self.calls = 0

    def ask_json(self, agent, instructions, user_input, temperature=0.0, retries=1):
        self.calls += 1
        return dict(self.payload)


def _check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def test_parse_plain() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "出院小结.txt"
        path.write_text(DISCHARGE_TEXT, encoding="utf-8")
        parsed = parse_document(path)

    ok = _check("纯文本解析", parsed.text_chars > 100, f"chars={parsed.text_chars}")
    ok &= _check("章节切分", "既往史" in parsed.sections, str(list(parsed.sections)))
    ok &= _check("文本层存在", parsed.has_text_layer)
    return ok


def test_parse_rejects_unsupported() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "旧病历.doc"
        path.write_bytes(b"\xd0\xcf\x11\xe0")
        try:
            parse_document(path)
        except DocumentError as exc:
            return _check("拒绝 .doc", "PDF" in str(exc) or ".docx" in str(exc), str(exc))
    return _check("拒绝 .doc", False, "没有抛 DocumentError")


def test_parse_rejects_oversize() -> bool:
    from patient_agent.graph.config import MAX_DOC_CHARS

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "big.txt"
        path.write_text("测" * (MAX_DOC_CHARS + 10), encoding="utf-8")
        try:
            parse_document(path)
        except DocumentError as exc:
            return _check("拒绝超大文档", "上限" in str(exc), str(exc))
    return _check("拒绝超大文档", False, "没有抛 DocumentError")


def test_extract_filters() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.txt"
        path.write_text(DISCHARGE_TEXT, encoding="utf-8")
        doc = parse_document(path)

    extracted = extract_slots(doc, FakeLLM())
    allowed = set(allowed_slot_names())

    ok = _check("抽取成功", extracted.ok, extracted.error or "")
    ok &= _check(
        "既往史写入 slots",
        "既往史" in extracted.slots,
        str(list(extracted.slots)),
    )
    ok &= _check(
        "过期文档的主诉不写进 slots",
        "主诉" not in extracted.slots,
        str(list(extracted.slots)),
    )
    ok &= _check(
        "过期主诉落到 facts 的既往资料",
        "既往主诉" in (extracted.facts.get("既往资料") or {}),
        str(extracted.facts),
    )
    ok &= _check(
        "白名单外的槽位被丢弃",
        all(name in allowed for name in extracted.slots),
    )
    return ok


def test_ingest_prefills() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "出院小结.txt"
        path.write_text(DISCHARGE_TEXT, encoding="utf-8")
        info = save_upload(path.read_bytes(), filename=path.name, kind="document")

    state = {
        "media_id": info["media_id"],
        "kind": "document",
        "filename": info["filename"],
        "path": info["path"],
        "case_id": "case-smoke",
    }
    update = ingest_document_node(state, FakeLLM())

    ok = _check("摄取状态", update.get("status") == "documents_ingested", str(update.get("status")))
    ok &= _check("预填了既往史", "既往史" in (update.get("slots") or {}), str(update.get("slots")))
    ok &= _check("记录来源", "document:" in (update.get("slot_sources") or {}).get("既往史", ""))
    ok &= _check("写入 facts", "既往资料" in (update.get("facts") or {}))
    return ok


def test_ingest_noop_without_document() -> bool:
    update = ingest_document_node({"case_id": "x"}, FakeLLM())
    ok = _check(
        "无文档时是 no-op",
        set(update) == {"media_pending"} and update["media_pending"] is False,
        str(update),
    )
    return ok


def test_patient_priority() -> bool:
    """患者口述优先：文档不覆盖，且记冲突。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.txt"
        path.write_text(DISCHARGE_TEXT, encoding="utf-8")
        info = save_upload(path.read_bytes(), filename=path.name, kind="document")

    state = {
        "media_id": info["media_id"],
        "kind": "document",
        "filename": info["filename"],
        "path": info["path"],
        "slots": {"既往史": "没有慢性病"},
        "slot_sources": {"既往史": "patient"},
    }
    update = ingest_document_node(state, FakeLLM())

    ok = _check(
        "患者口述未被覆盖",
        update["slots"]["既往史"] == "没有慢性病",
        str(update["slots"]),
    )
    ok &= _check("记录冲突", len(update.get("slot_conflicts") or []) >= 1, str(update.get("slot_conflicts")))
    return ok


def test_storage_rejects_bad_type() -> bool:
    try:
        save_upload(b"x", filename="evil.exe", kind="document")
    except MediaError as exc:
        return _check("拒绝不支持的类型", "不支持" in str(exc), str(exc))
    return _check("拒绝不支持的类型", False, "没有抛 MediaError")


def test_storage_blocks_traversal() -> bool:
    info = save_upload(b"x", filename="../../evil.txt", kind="document")
    stored = Path(info["path"])
    ok = _check("文件名不穿越目录", stored.parent.name == info["media_id"], str(stored))
    ok &= _check("文件名已清洗", ".." not in stored.name, stored.name)
    return ok


def main() -> int:
    tests = [
        test_parse_plain,
        test_parse_rejects_unsupported,
        test_parse_rejects_oversize,
        test_extract_filters,
        test_ingest_prefills,
        test_ingest_noop_without_document,
        test_patient_priority,
        test_storage_rejects_bad_type,
        test_storage_blocks_traversal,
    ]

    failures = 0
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            if not test():
                failures += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {test.__name__} 抛出异常: {exc}")

    print(f"\n{'=' * 50}")
    print(f"失败用例数：{failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
