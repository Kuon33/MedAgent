"""文档解析：把 PDF / DOCX / 文本文件解析成结构化文本。

纯 Python 实现，不调用任何模型。输出是一个 ParsedDoc，交给抽取层用模型转成槽位。

支持：
- .pdf   —— pymupdf 抽文本层 + 表格；无文本层时标记 needs_ocr
- .docx  —— python-docx 抽段落与表格
- .txt/.md —— 直接读文本

降级策略见 `parse_document` 的 docstring。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from patient_agent.graph.config import (
    ALLOWED_DOC_SUFFIXES,
    MAX_DOC_CHARS,
    MAX_DOC_PAGES,
    SCAN_PAGE_CHAR_THRESHOLD,
)


class DocumentError(Exception):
    """文档无法解析。message 直接给用户看。"""


@dataclass
class ParsedDoc:
    """解析结果。"""

    filename: str
    suffix: str
    text: str = ""
    pages: int = 0
    tables: list[str] = field(default_factory=list)
    has_text_layer: bool = True
    needs_ocr: bool = False
    sections: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def text_chars(self) -> int:
        return len(self.text)


# 病历里常见的章节标题。用来切分段落。
SECTION_PATTERNS = (
    "主诉",
    "现病史",
    "既往史",
    "个人史",
    "婚育史",
    "家族史",
    "过敏史",
    "用药史",
    "既往用药",
    "体格检查",
    "查体",
    "辅助检查",
    "辅助检查资料",
    "初步诊断",
    "诊断",
    "入院诊断",
    "出院诊断",
    "治疗经过",
    "诊疗经过",
    "治疗方案",
    "出院医嘱",
    "医嘱",
    "手术史",
    "手术史及外伤史",
    "疾病史",
    "备注",
)

_SECTION_RE = re.compile(
    r"^\s*(?:[一二三四五六七八九十]+[、.．]|\d+[、.．）)]|\(\d+\)|（\d+）)?\s*"
    r"(" + "|".join(re.escape(name) for name in SECTION_PATTERNS) + r")\s*[:：]?\s*(.*)$"
)


def _split_sections(text: str) -> dict[str, str]:
    """按病历常见章节标题切分文本。

    兼容两种写法：
    - 标题独占一行（内容在后面的行）
    - 标题和内容在同一行（`既往史：高血压病史 10 年`）

    切不出来就返回空字典（调用方按需取全文）。
    """
    sections: dict[str, list[str]] = {}
    current_key: str | None = None

    for line in text.splitlines():
        match = _SECTION_RE.match(line)
        if match:
            current_key = match.group(1)
            sections.setdefault(current_key, [])
            inline = (match.group(2) or "").strip()
            if inline:
                sections[current_key].append(inline)
            continue
        if current_key is not None:
            sections[current_key].append(line)

    return {key: "\n".join(lines).strip() for key, lines in sections.items() if lines}


def _parse_pdf(path: Path) -> ParsedDoc:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover
        raise DocumentError("未安装 pymupdf，无法解析 PDF") from exc

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        # pymupdf 对加密文档抛的异常类型不固定，统一转成可读提示
        text = str(exc).lower()
        if "password" in text or "encrypt" in text:
            raise DocumentError("该 PDF 已加密，请上传无密码的 PDF") from exc
        raise DocumentError(f"PDF 打开失败：{exc}") from exc

    with doc:
        if doc.needs_pass:
            raise DocumentError("该 PDF 已加密，请上传无密码的 PDF")

        pages = doc.page_count
        if pages > MAX_DOC_PAGES:
            raise DocumentError(
                f"文档共 {pages} 页，超过上限 {MAX_DOC_PAGES} 页，请拆分后再上传"
            )

        texts: list[str] = []
        tables: list[str] = []
        total_chars = 0

        for index in range(pages):
            page = doc.load_page(index)
            page_text = page.get_text("text") or ""
            total_chars += len(page_text.strip())
            texts.append(page_text)

            try:
                found = page.find_tables()
            except Exception:
                found = []
            for table in found:
                try:
                    rows = table.extract()
                except Exception:
                    continue
                if not rows:
                    continue
                lines = []
                for row in rows:
                    cells = [str(cell or "").strip().replace("\n", " ") for cell in row]
                    if any(cells):
                        lines.append(" | ".join(cells))
                if lines:
                    tables.append(f"[第{index + 1}页表格]\n" + "\n".join(lines))

    text = "\n".join(texts).strip()

    # 没有文本层：整篇几乎没字 → 判定为扫描件
    has_text_layer = total_chars >= max(1, pages) * SCAN_PAGE_CHAR_THRESHOLD
    if not has_text_layer:
        return ParsedDoc(
            filename=path.name,
            suffix=path.suffix.lower(),
            text=text,
            pages=pages,
            tables=tables,
            has_text_layer=False,
            needs_ocr=True,
            error=(
                "这份文档看起来是扫描件，没有可提取的文字。"
                "识别质量有限，建议补充文字描述。"
            ),
        )

    if total_chars > MAX_DOC_CHARS:
        raise DocumentError(
            f"文档约 {total_chars} 字，超过上限 {MAX_DOC_CHARS} 字，请拆分后再上传"
        )

    return ParsedDoc(
        filename=path.name,
        suffix=path.suffix.lower(),
        text=text,
        pages=pages,
        tables=tables,
        has_text_layer=True,
        sections=_split_sections(text),
    )


def _parse_docx(path: Path) -> ParsedDoc:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise DocumentError("未安装 python-docx，无法解析 Word 文档") from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise DocumentError(f"Word 文档打开失败：{exc}") from exc

    paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    text = "\n".join(paragraphs)

    tables: list[str] = []
    for index, table in enumerate(document.tables):
        lines = []
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))
        if lines:
            tables.append(f"[表格{index + 1}]\n" + "\n".join(lines))

    if len(text) > MAX_DOC_CHARS:
        raise DocumentError(
            f"文档约 {len(text)} 字，超过上限 {MAX_DOC_CHARS} 字，请拆分后再上传"
        )

    return ParsedDoc(
        filename=path.name,
        suffix=path.suffix.lower(),
        text=text,
        pages=1,
        tables=tables,
        has_text_layer=True,
        sections=_split_sections(text),
    )


def _parse_plain(path: Path) -> ParsedDoc:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) > MAX_DOC_CHARS:
        raise DocumentError(
            f"文档约 {len(text)} 字，超过上限 {MAX_DOC_CHARS} 字，请拆分后再上传"
        )
    return ParsedDoc(
        filename=path.name,
        suffix=path.suffix.lower(),
        text=text,
        pages=1,
        has_text_layer=True,
        sections=_split_sections(text),
    )


def parse_document(path: str | Path) -> ParsedDoc:
    """解析文档为 ParsedDoc。

    失败时抛 DocumentError，message 是给用户看的提示。
    旧版 .doc / .wps 等格式不支持，会明确拒绝而不是尝试转换。
    """
    path = Path(path)
    if not path.exists():
        raise DocumentError("文件不存在")

    suffix = path.suffix.lower()

    if suffix in {".doc", ".wps", ".rtf", ".odt"}:
        raise DocumentError(
            f"暂不支持 {suffix} 格式，请另存为无密码的 PDF 或 .docx 后再上传"
        )
    if suffix not in ALLOWED_DOC_SUFFIXES:
        allowed = "、".join(sorted(ALLOWED_DOC_SUFFIXES))
        raise DocumentError(f"不支持的文件类型 {suffix or '未知'}，请上传 {allowed}")

    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix == ".docx":
        return _parse_docx(path)
    return _parse_plain(path)


__all__ = ["DocumentError", "ParsedDoc", "parse_document"]
