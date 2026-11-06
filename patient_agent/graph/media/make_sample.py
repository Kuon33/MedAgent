"""生成一份用于测试的 PDF 病历（不依赖外部数据）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

CONTENT = """出院小结

姓名：张某某  性别：男  年龄：62岁
入院日期：2025-08-13  出院日期：2025-08-20

主诉：反复胸闷 3 天
现病史：患者 3 天前无明显诱因出现胸闷，位于胸骨后，持续不缓解，伴出汗，无明显呼吸困难，无恶心呕吐。
既往史：高血压病史 10 年，2 型糖尿病 5 年。
用药史：苯磺酸氨氯地平 5mg 每日一次；二甲双胍 0.5g 每日两次。
过敏史：青霉素过敏。
体格检查：血压 150/95 mmHg，心率 88 次/分，BMI 26.4。
辅助检查：心电图示窦性心律，ST 段压低 0.1mV；空腹血糖 8.2 mmol/L。
初步诊断：高血压 2 级极高危；2 型糖尿病。
出院医嘱：低盐低脂饮食，监测血压，门诊随诊。
"""


def build_pdf(target: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(40, 40, 555, 800),
        CONTENT,
        fontsize=11,
        fontname="china-s",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(target))
    doc.close()
    return target


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/samples/出院小结.pdf")
    path = build_pdf(out)
    print(f"已生成：{path}")
