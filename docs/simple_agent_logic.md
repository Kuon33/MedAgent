# 极简预问诊 Agent 逻辑说明

本文档说明 `patient_agent/simple_agent` 的当前设计。它不使用前端、不使用 LangGraph、不使用数据库，先用最短路径跑通核心业务。

```text
患者输入 -> 主分诊 Agent -> 科室 Skill 子 Agent -> 可回退重分诊 -> 报告生成 -> 写入患者记忆 -> 可选一键挂号
```

## 目标

系统用于患者线上挂号前的预问诊。主 Agent 根据患者描述、历史摘要和科室 skill 判断推荐科室；随后进入对应科室的预问诊子 Agent。子 Agent 如果发现科室不匹配，会生成 handoff 摘要并回退给主 Agent。科室切换最多两次。最终系统生成给医生看的预问诊报告，并可模拟一键挂号。

## 当前技术栈

```text
Python
OpenAI SDK
Pydantic
JSON 文件存储
Markdown 科室 Skill
JSONL 运行日志
```

当前不包含前端、LangGraph、数据库、向量库和任务队列。

## 文件结构

```text
patient_agent/simple_agent/
  app.py              命令行入口和主循环
  openai_client.py    OpenAI SDK 封装，支持 Responses 和 Chat Completions
  state.py            PatientMemory、CaseState、AgentResult
  memory.py           患者长期记忆读写和本次问诊摘要
  skills.py           科室 skill 加载和简单关键词检索
  triage.py           主分诊 Agent
  specialty.py        科室子 Agent
  reporting.py        报告生成
  observability.py    JSONL 运行日志
  prompts.py          主 Agent、子 Agent、报告 prompt

skills/departments/
  cardiology.md
  respiratory.md
  gastroenterology.md
  dermatology.md
  neurology.md

data/
  patients/           患者长期摘要
  cases/              本次问诊状态
  reports/            Markdown 报告
  runs/               JSONL 调用日志
```

## 运行配置

`.env` 示例：

```bash
OPENAI_API_KEY=你的 API key
OPENAI_MODEL=deepseek-ai/DeepSeek-V3.2
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_API_MODE=chat_completions
PATIENT_AGENT_MAX_SWITCHES=2
PATIENT_AGENT_MAX_SPECIALTY_TURNS=4
```

如果使用官方 OpenAI，可以把 `OPENAI_BASE_URL` 留空，并把 `OPENAI_API_MODE` 设置为 `auto` 或 `responses`。

如果使用 OpenAI-compatible 服务、代理地址或本地模型服务，一般设置：

```bash
OPENAI_BASE_URL=https://api.example.com/v1
OPENAI_API_MODE=chat_completions
```

也可以使用 `OPENAI_MODEL_URL`，代码会把它当作 `OPENAI_BASE_URL` 的别名。

## 核心状态

### PatientMemory

患者长期记忆，用 `patient_id` 锚定，保存到 `data/patients/{patient_id}.json`。

```text
patient_id
profile_summary
history_summaries
updated_at
```

### CaseState

本次预问诊状态，保存到 `data/cases/{case_id}.json`。

```text
case_id
patient_id
messages
facts
current_department
department_attempts
switch_count
red_flags
status
report_path
appointment
created_at
updated_at
```

### AgentResult

主 Agent 和子 Agent 都输出：

```text
action
reply
data
```

`action` 控制流程：

```text
ask_more          继续追问患者
triage_done       主 Agent 已判断科室
specialty_done    科室子 Agent 问诊完成
handoff           子 Agent 判断当前科室不合适
urgent            出现急诊风险
finish            流程结束
```

## 主循环

入口在 `patient_agent/simple_agent/app.py` 的 `run_case()`。

```text
1. 输入 patient_id
2. 读取 PatientMemory
3. 创建 CaseState
4. 患者描述主要不适
5. 主 Agent 分诊
6. 信息不足时继续追问
7. 急诊风险时生成报告并结束
8. 判断出科室后进入科室子 Agent
9. 子 Agent 继续科室预问诊
10. 子 Agent handoff 时回到主 Agent
11. 达到切换上限时建议线下综合评估
12. 子 Agent 完成后生成报告
13. 报告只保存到文件，不打印医生报告全文
14. 询问患者是否需要一键挂号
15. 保存 case，压缩摘要写入 PatientMemory
16. 写入运行日志
```

## 科室问诊收束

科室子 Agent 应尽快问清关键症状，不能无休止追问。当前通过两层控制：

```text
1. prompt 和 skill 明确要求优先问关键问题，信息足够就 specialty_done
2. 代码层使用 PATIENT_AGENT_MAX_SPECIALTY_TURNS 限制每个科室最多追问轮数，默认 4 轮
```

## 主 Agent 逻辑

入口：

```text
triage.py -> run_triage_agent()
```

输入包括患者长期摘要、最近历史问诊摘要、当前 CaseState、本地科室关键词命中和所有科室 Skill 摘要。

输出 JSON：

```json
{
  "action": "ask_more | triage_done | urgent",
  "reply": "给患者看的回复",
  "department": "推荐科室，没有则为空",
  "reason": "判断原因",
  "red_flags": ["红旗风险"],
  "facts_patch": {"主诉": "...", "持续时间": "...", "严重程度": "..."}
}
```

## 科室子 Agent 逻辑

入口：

```text
specialty.py -> run_specialty_agent()
```

子 Agent 读取当前 `current_department` 对应的 skill。

输出 JSON：

```json
{
  "action": "ask_more | specialty_done | handoff | urgent",
  "reply": "给患者看的回复",
  "reason": "判断原因",
  "summary": "患者病情摘要",
  "suggested_department": "需要转科时填写，否则为空",
  "facts_patch": {"补充信息": "..."},
  "red_flags": ["红旗风险"],
  "report_notes": "给医生看的问诊要点"
}
```

## 科室切换

切换逻辑在：

```text
specialty.py -> apply_handoff()
```

每次 handoff 会记录 `department`、`reason`、`summary` 和 `created_at`，并让 `switch_count += 1`。默认最多切换两次。

## 一键挂号

报告生成后，命令行会询问：

```text
是否需要一键挂号到{科室}？输入 y/yes/是 确认：
```

如果患者输入 `y`、`yes` 或 `是`，系统会模拟医院挂号成功，并把预约号、科室和状态写入当前 case 的 `appointment` 字段。

## 报告生成

入口：

```text
reporting.py -> generate_report()
```

报告保存到：

```text
data/reports/{case_id}.md
```

医生报告不会直接打印在命令行里，命令行只提示保存路径。

## 记忆系统

每次流程结束后，`memory.py -> append_case_summary()` 会调用模型把本次问诊压缩为长期摘要，并追加到：

```text
data/patients/{patient_id}.json
```

## 观测日志

路径：

```text
data/runs/{run_id}.jsonl
```

记录 `case_started`、`model_call`、`triage_result`、`specialty_result` 和 `case_finished`。
