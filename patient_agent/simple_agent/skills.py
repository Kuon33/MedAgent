from __future__ import annotations

from pathlib import Path

from patient_agent.simple_agent.config import SKILLS_ROOT


DEPARTMENT_FILES = {
    "心内科": "cardiology.md",
    "呼吸内科": "respiratory.md",
    "消化内科": "gastroenterology.md",
    "皮肤科": "dermatology.md",
    "神经内科": "neurology.md",
}

DEPARTMENT_KEYWORDS = {
    "心内科": ["胸痛", "胸闷", "心悸", "血压", "心前区", "出汗", "冠心病"],
    "呼吸内科": ["咳嗽", "咳痰", "发热", "气短", "呼吸困难", "喘", "肺"],
    "消化内科": ["腹痛", "胃痛", "反酸", "烧心", "恶心", "呕吐", "腹泻"],
    "皮肤科": ["皮疹", "瘙痒", "红斑", "水疱", "过敏", "皮肤"],
    "神经内科": ["头痛", "头晕", "麻木", "无力", "抽搐", "意识", "偏瘫"],
}


def normalize_department(name: str | None) -> str | None:
    if not name:
        return None
    text = name.strip()
    if text in DEPARTMENT_FILES:
        return text
    for department in DEPARTMENT_FILES:
        if department in text or text in department:
            return department
    return text


def load_skill(department: str) -> str:
    normalized = normalize_department(department) or department
    filename = DEPARTMENT_FILES.get(normalized)
    if not filename:
        raise FileNotFoundError(f"unsupported department skill: {department}")
    path = SKILLS_ROOT / filename
    return path.read_text(encoding="utf-8")


def load_all_skill_summaries() -> str:
    blocks = []
    for department, filename in DEPARTMENT_FILES.items():
        path = SKILLS_ROOT / filename
        if path.exists():
            text = path.read_text(encoding="utf-8")
            blocks.append(f"## {department}\n{text[:1200]}")
    return "\n\n".join(blocks)


def simple_skill_search(text: str, top_k: int = 3) -> list[dict]:
    scores = []
    for department, keywords in DEPARTMENT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score:
            scores.append({"department": department, "score": score, "keywords": keywords})
    return sorted(scores, key=lambda item: item["score"], reverse=True)[:top_k]

