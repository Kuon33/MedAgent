"""科室结构化规格：必填槽位、采集重点、急诊红旗规则。

分工说明：
- Markdown Skill（``skills/departments/*.md``）仍然是把给模型看的问诊正文；
- 本模块保存**机器可读**的部分：必填槽位、采集重点、红旗正则。

为什么红旗要放在代码里：模型的判断不稳定，而「是否急诊」是安全底线。
这里用正则 + 否定词处理做兜底，命中即可推翻模型的乐观结论。

后续可以把 ``required_slots`` / ``red_flag_rules`` 迁到 Markdown 的 YAML
front matter，改动只涉及本模块的解析函数。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from patient_agent.graph.config import SKILLS_ROOT


DEPARTMENT_FILES = {
    "心内科": "cardiology.md",
    "呼吸内科": "respiratory.md",
    "消化内科": "gastroenterology.md",
    "皮肤科": "dermatology.md",
    "神经内科": "neurology.md",
}

DEPARTMENT_ALIASES = {
    "心血管内科": "心内科",
    "心血管": "心内科",
    "心内": "心内科",
    "呼吸科": "呼吸内科",
    "呼吸内": "呼吸内科",
    "消化科": "消化内科",
    "消化内": "消化内科",
    "神经科": "神经内科",
    "神经内": "神经内科",
    "皮肤": "皮肤科",
}

SUPPORTED_DEPARTMENTS = tuple(DEPARTMENT_FILES)

# 还没定科室时（通用采集）需要覆盖的槽位。
GENERAL_SLOTS = ("主诉", "病程", "伴随症状", "既往史", "用药史", "过敏史", "危险信号")


@dataclass(frozen=True)
class RedFlagRule:
    """一条急诊红旗规则。"""

    id: str
    patterns: tuple[str, ...]
    severity: str  # urgent | warning
    message: str


@dataclass(frozen=True)
class DepartmentSpec:
    """科室的机器可读规格。"""

    name: str
    file: str
    required_slots: tuple[str, ...]
    focus: str
    red_flags: tuple[RedFlagRule, ...] = ()


GENERAL_RED_FLAGS = (
    RedFlagRule(
        "consciousness",
        (r"意识(?:不清|模糊|丧失)", r"昏迷", r"晕厥"),
        "urgent",
        "意识障碍或晕厥",
    ),
    RedFlagRule(
        "dyspnea",
        (r"呼吸(?:困难|急促|费力)", r"喘不上气", r"窒息感"),
        "urgent",
        "明显呼吸困难",
    ),
    RedFlagRule(
        "stroke",
        (
            r"(?:一侧|单侧|左边|右边)[^，。；]{0,4}(?:无力|麻木|不能动)",
            r"口角歪斜",
            r"(?:言语|说话)不清",
        ),
        "urgent",
        "卒中样表现",
    ),
    RedFlagRule(
        "bleeding",
        (r"呕血", r"咯血", r"大量出血", r"便血", r"黑便"),
        "urgent",
        "活动性出血",
    ),
    RedFlagRule(
        "severe_pain",
        (r"(?:剧烈|难以忍受|爆炸样)[^，。；]{0,4}(?:头痛|腹痛|胸痛)",),
        "urgent",
        "剧烈疼痛",
    ),
    RedFlagRule(
        "high_fever",
        (r"(?:39|40)(?:\.\d)?\s*度", r"高热"),
        "warning",
        "高热",
    ),
)


DEPARTMENT_SPECS: dict[str, DepartmentSpec] = {
    "心内科": DepartmentSpec(
        name="心内科",
        file="cardiology.md",
        required_slots=(
            "主诉",
            "病程",
            "部位",
            "性质",
            "放射",
            "诱因缓解",
            "伴随症状",
            "既往史",
            "用药史",
            "过敏史",
        ),
        focus="胸痛/胸闷的部位、性质、持续时间、放射、诱因与缓解，是否伴大汗、呼吸困难、晕厥，心血管危险因素",
        red_flags=(
            RedFlagRule(
                "acs",
                (
                    r"(?:持续|压榨样?|紧缩样)[^，。；]{0,6}胸(?:痛|闷)",
                    r"胸(?:痛|闷)[^，。；]{0,6}(?:大汗|出汗|濒死感)",
                    r"(?:胸闷|胸痛)[^，。；]{0,6}晕厥",
                ),
                "urgent",
                "持续或压榨样胸痛，或伴大汗、晕厥",
            ),
            RedFlagRule(
                "palpitation_syncope",
                (r"(?:心悸|心慌)[^，。；]{0,6}(?:晕厥|黑蒙|黑矇)",),
                "urgent",
                "心悸伴晕厥或黑蒙",
            ),
        ),
    ),
    "呼吸内科": DepartmentSpec(
        name="呼吸内科",
        file="respiratory.md",
        required_slots=(
            "主诉",
            "病程",
            "咳痰性状",
            "发热",
            "呼吸困难",
            "胸痛",
            "既往史",
            "用药史",
            "过敏史",
        ),
        focus="咳嗽咳痰的性状与病程、发热程度、是否有呼吸困难或咯血、胸痛与咳嗽的关系、既往呼吸道病史",
        red_flags=(
            RedFlagRule(
                "severe_respiratory",
                (
                    r"(?:呼吸困难|气短)[^，。；]{0,4}(?:加重|不能平卧)",
                    r"口唇(?:发紫|紫绀)",
                ),
                "urgent",
                "呼吸困难加重或紫绀",
            ),
            RedFlagRule("hemoptysis", (r"咯血",), "urgent", "咯血"),
        ),
    ),
    "消化内科": DepartmentSpec(
        name="消化内科",
        file="gastroenterology.md",
        required_slots=(
            "主诉",
            "病程",
            "部位",
            "与进食关系",
            "反酸烧心",
            "呕吐黑便",
            "既往史",
            "用药史",
            "过敏史",
        ),
        focus="腹痛部位与进食关系、反酸烧心、有无呕血黑便、呕吐、既往胃肠病史与用药（尤其止痛药）",
        red_flags=(
            RedFlagRule("gi_bleeding", (r"呕血", r"黑便", r"便血"), "urgent", "消化道出血表现"),
            RedFlagRule(
                "acute_abdomen",
                (r"(?:剧烈|持续)[^，。；]{0,4}腹痛", r"腹部[^，。；]{0,3}(?:板状|僵硬)"),
                "urgent",
                "剧烈或持续腹痛",
            ),
        ),
    ),
    "皮肤科": DepartmentSpec(
        name="皮肤科",
        file="dermatology.md",
        required_slots=(
            "主诉",
            "病程",
            "部位",
            "形态",
            "诱因",
            "瘙痒程度",
            "全身症状",
            "过敏史",
            "用药史",
        ),
        focus="皮损部位与形态、起病诱因（新用药/接触物）、瘙痒程度、有无全身症状或黏膜受累",
        red_flags=(
            RedFlagRule(
                "anaphylaxis",
                (r"喉头水肿", r"嘴唇?肿", r"呼吸(?:困难|费力)", r"窒息感"),
                "urgent",
                "疑似严重过敏反应",
            ),
            RedFlagRule(
                "mucosal_involvement",
                (r"(?:口腔|眼部|生殖器)[^，。；]{0,4}(?:糜烂|溃疡)", r"皮肤[^，。；]{0,4}大面积[^，。；]{0,4}松解"),
                "urgent",
                "黏膜受累或大面积皮肤松解",
            ),
        ),
    ),
    "神经内科": DepartmentSpec(
        name="神经内科",
        file="neurology.md",
        required_slots=(
            "主诉",
            "起病方式",
            "病程",
            "部位性质",
            "伴随症状",
            "肢体无力言语",
            "意识抽搐",
            "既往史",
            "用药史",
        ),
        focus="起病是突发还是渐进、头痛部位性质、有无肢体无力/言语不清/意识丧失/抽搐、既往卒中或癫痫史",
        red_flags=(
            RedFlagRule(
                "thunderclap_headache",
                (r"(?:突发|突然)[^，。；]{0,4}(?:剧烈|爆炸样)?头痛", r"爆炸样头痛"),
                "urgent",
                "突发剧烈头痛",
            ),
            RedFlagRule(
                "focal_deficit",
                (r"(?:肢体|手脚)[^，。；]{0,4}(?:无力|麻木|不能动)", r"口角歪斜", r"(?:言语|说话)不清"),
                "urgent",
                "局灶性神经功能缺损",
            ),
        ),
    ),
}


# 否定词与子句边界：判断「没有持续胸痛」不能算命中「持续胸痛」。
NEGATORS = ("没有", "未见", "否认", "不曾", "无", "未", "不", "非", "排除")
CLAUSE_BREAKS = "，。；！？,.;!?\n"


def normalize_department(name: str | None) -> str | None:
    """把任意写法归一到受支持的科室名；不支持的科室返回 None。

    这一点很关键：旧版本会把「骨科」「急诊科」原样返回，随后 load_skill
    抛 FileNotFoundError，整个图直接崩掉。
    """
    if not name:
        return None
    text = str(name).strip()
    if not text:
        return None
    for canonical in DEPARTMENT_FILES:
        if canonical in text:
            return canonical
    for alias, canonical in DEPARTMENT_ALIASES.items():
        if alias in text:
            return canonical
    return None


def department_spec(name: str | None) -> DepartmentSpec | None:
    canonical = normalize_department(name)
    return DEPARTMENT_SPECS.get(canonical) if canonical else None


def general_spec() -> DepartmentSpec:
    return DepartmentSpec(
        name="通用",
        file="",
        required_slots=GENERAL_SLOTS,
        focus="主诉、病程、伴随症状、既往史、用药史、过敏史、危险信号",
        red_flags=(),
    )


@lru_cache(maxsize=8)
def load_skill_body(department: str | None) -> str:
    """读取科室 Markdown skill 正文；读取失败返回空串而不是抛异常。"""
    spec = department_spec(department)
    if not spec or not spec.file:
        return ""
    path = SKILLS_ROOT / spec.file
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _clause_start(text: str, position: int) -> int:
    start = 0
    for index in range(position - 1, -1, -1):
        if text[index] in CLAUSE_BREAKS:
            start = index + 1
            break
    return start


def _is_negated(text: str, position: int) -> bool:
    clause = text[_clause_start(text, position):position]
    return any(negator in clause for negator in NEGATORS)


def match_red_flags(text: str, department: str | None = None) -> list[RedFlagRule]:
    """在文本中匹配红旗规则，处理否定词，按规则 id 去重。"""
    if not text:
        return []

    rules: list[RedFlagRule] = list(GENERAL_RED_FLAGS)
    spec = department_spec(department)
    if spec:
        rules.extend(spec.red_flags)

    hits: dict[str, RedFlagRule] = {}
    for rule in rules:
        for pattern in rule.patterns:
            for match in re.finditer(pattern, text):
                if not _is_negated(text, match.start()):
                    hits[rule.id] = rule
                    break
            if rule.id in hits:
                break
    return list(hits.values())


__all__ = [
    "DEPARTMENT_FILES",
    "DEPARTMENT_SPECS",
    "GENERAL_SLOTS",
    "SUPPORTED_DEPARTMENTS",
    "DepartmentSpec",
    "RedFlagRule",
    "department_spec",
    "general_spec",
    "load_skill_body",
    "match_red_flags",
    "normalize_department",
]
