# 预问诊测试 Case

这些 case 用于人工回归测试当前 CLI agent。每个 case 都覆盖一个目标科室，并额外包含急诊红旗和科室切换场景。

运行方式：

```bash
python -m patient_agent.simple_agent.app --patient-id test_cardiology_001
```

启动后按对应 case 的 `suggested_inputs` 顺序回答。模型可能会用不同方式追问，回答时保持同一患者设定即可。

## 通用检查点

- 命令行最后只提示医生报告已保存，不打印完整医生报告。
- `data/reports/{case_id}.md` 应生成报告。
- `data/cases/{case_id}.json` 应保存当前 case 状态。
- 如果最后输入 `是`、`y` 或 `yes`，应显示挂号成功，并在 case 的 `appointment` 字段写入预约信息。
- 科室预问诊不应超过 `.env` 中的 `PATIENT_AGENT_MAX_SPECIALTY_TURNS`。

## Case 列表

| id | 目标科室 | 重点 |
| --- | --- | --- |
| test_cardiology_001 | 心内科 | 活动后胸闷、短暂胸痛、高血压史 |
| test_respiratory_001 | 呼吸内科 | 咳嗽咳痰、低热、无明显呼吸困难 |
| test_gastroenterology_001 | 消化内科 | 上腹痛、反酸烧心、餐后加重 |
| test_dermatology_001 | 皮肤科 | 皮疹瘙痒、疑似接触诱因、无喉头水肿 |
| test_neurology_001 | 神经内科 | 反复头痛头晕、无卒中红旗 |
| test_urgent_001 | 急诊红旗 | 突发剧烈头痛、一侧无力、言语不清 |
| test_handoff_001 | 科室切换 | 初始像心内科，后续信息更像呼吸内科 |

## 边界提醒

当前项目只有 5 个科室 skill。如果输入明显属于骨科、眼科、耳鼻喉科等未覆盖科室，理想行为应是继续追问、建议线下综合评估或转全科，而不是进入未配置 skill。
