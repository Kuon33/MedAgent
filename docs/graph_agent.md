# LangGraph 版预问诊 Agent

本目录是原 `simple_agent` 的 LangGraph 重构版，**原 CLI 保持不变、仍可正常使用**。

## 架构：双 Agent 分工

```text
患者输入
   ↓
┌─────────────────────────┐
│  采集智能体 collect_node  │  只补信息，禁止判断科室
│  按「必填槽位」逐项问全   │  红旗由代码正则兜底
└─────────────────────────┘
   ↓ collection_done
┌─────────────────────────┐
│  分诊智能体 triage_node   │  只判断科室与紧急程度
│  只看结构化 slots，不看对话 │  禁止给诊断
└─────────────────────────┘
   ↓ 选定科室
┌─────────────────────────┐
│  科室采集 collect_node    │  按该科室的 required_slots 定向补齐
└─────────────────────────┘
   ↓
报告 → 一键挂号 → 写入患者记忆
```

两个 Agent 的 prompt 在 `core/prompts.py`，机器可读的科室规格（必填槽位、红旗正则）在 `core/departments.py`。

## 目录

```text
patient_agent/graph/
  config.py           独立配置（.env 加载、路径、轮数阈值）
  state.py            PatientState 定义与 reducer
  routes.py           条件边路由函数（同步/异步图共用）
  nodes.py            通用节点（ask_patient / report / appointment / booking / memory）
  graph.py            同步图（CLI / Gradio 用）
  async_graph.py      异步图（SSE API 用）
  cli.py              CLI 入口
  server.py           FastAPI + SSE 服务
  gradio_app.py       Gradio 聊天界面
  smoke.py            CLI 端到端冒烟测试
  api_smoke.py        SSE 接口端到端冒烟测试
  gradio_smoke.py     Gradio 界面逻辑端到端测试
  agents/
    collect.py        采集智能体
    triage.py         分诊智能体
  core/
    departments.py    科室规格：必填槽位、采集重点、急诊红旗正则
    llm.py            GraphLLM（支持 json_mode、失败重试、延迟创建 client）
    prompts.py        COLLECT_PROMPT / TRIAGE_PROMPT / DOC_EXTRACT_PROMPT
    utils.py          prompt 拼装小工具
  media/
    documents.py      文档解析：PDF / DOCX / 纯文本（纯 Python，不调模型）
    extract.py        模型抽取结构化槽位（一次调用，guided JSON）
    ingest.py         合并进状态（预填 slots，附冲突检测）
    storage.py        上传文件落盘、文件名清洗
    asr.py            语音识别（本地 Qwen3-ASR-1.7B），音频预处理 + 转写
    smoke.py          文档摄取冒烟测试（不依赖模型）
    asr_smoke.py      语音识别冒烟测试（不依赖模型）
    compare.py        带/不带文档的采集轮次对比
    make_sample.py    生成样例病历 PDF
```

## 图结构

```mermaid
graph TD
    start([__start__]) --> ingest_document
    ingest_document --> collect
    collect -. 未收束 .-> ask_patient
    collect -. 收束 .-> triage
    collect -. 急症 .-> report
    ask_patient -. 上传了病历 .-> ingest_document
    ask_patient -. 未定科室 .-> collect
    ask_patient -. 已定科室 .-> specialty_collect
    triage -. 缺信息 .-> collect
    triage -. 换科室 .-> specialty_collect
    triage -. 收束 .-> report
    specialty_collect -. 未收束 .-> ask_patient
    specialty_collect -. 收束 .-> report
    report --> appointment
    appointment --> booking
    booking --> memory
    memory --> end([__end__])
```

## 与 simple_agent 的对应关系

| simple_agent | LangGraph 版 |
| --- | --- |
| `run_triage_agent()` | `agents/triage.py` → `triage_node` |
| `run_specialty_agent()` | `agents/collect.py` → `collect_node`（科室阶段） |
| `input()` 阻塞等待 | `nodes.py` → `ask_patient_node` 中的 `interrupt()` |
| `finish_with_report()` | `nodes.py` → `report_node` |
| `offer_appointment()` | `appointment_node`（interrupt）+ `booking_node` |
| `append_case_summary()` | `memory_node` |
| `case.merge_facts()` | state reducer `merge_dict` |
| `case.merge_red_flags()` | state reducer `merge_unique` |
| `data/cases/*.json` | LangGraph checkpointer（`data/checkpoints.sqlite`） |
| `data/patients/*.json` | 仍复用 `simple_agent.memory` |
| `data/reports/*.md` | 仍复用 `simple_agent.reporting` |

## 运行 CLI

```powershell
# 需要在项目根目录执行
python -m patient_agent.graph.cli --patient-id patient_001

# 也可直接给首句，非交互
python -m patient_agent.graph.cli --patient-id patient_001 --input "最近一周爬楼后胸口发闷"
```

## 运行 API

```powershell
python -m patient_agent.graph.server
# 或
uvicorn patient_agent.graph.server:app --reload --port 8000
```

打开 http://127.0.0.1:8000/docs 可看到自动生成的 OpenAPI 文档。

### 同时挂载 Gradio 界面（同一端口）

```powershell
python -m patient_agent.graph.server --with-gradio
```

然后访问：

- http://127.0.0.1:8000/ —— Gradio 聊天界面（挂在根路径）
- http://127.0.0.1:8000/docs —— OpenAPI 文档
- http://127.0.0.1:8000/api/consult/stream —— SSE 接口

> ⚠️ Gradio 必须挂在**根路径**。挂在子路径（如 `/ui`）时，
> 页面配置里的 `api_prefix` 会丢掉子路径前缀，前端请求
> `/gradio_api/info` 会 404，界面能打开但无法交互。
> 业务 API 都在 `/api/*` 下，与根路径不冲突。

也可以让 Gradio 独立运行（默认 7860 端口）：

```powershell
python -m patient_agent.graph.gradio_app
```

## Gradio 界面

`patient_agent/graph/gradio_app.py` 把同一张 LangGraph 图包装成聊天界面：

```text
┌──────────────────────┬────────────────────┐
│  问诊对话 (Chatbot)   │  医生预问诊报告      │
│                      │  (Markdown)         │
│  [输入框] [发送]      │                    │
│  [重新问诊]           │  挂号信息            │
└──────────────────────┴────────────────────┘
```

实现要点：

- 使用**同步** `SqliteSaver`（与 CLI 一致），不用 `AsyncSqliteSaver`。
  Gradio 事件在独立线程里跑，sqlite 连接是 `check_same_thread=False`，跨线程安全。
- `thread_id` 存在 `gr.State` 里，跨 submit 保持不变，
  用于恢复 LangGraph 的 `interrupt` 挂起点。
- 首次 submit 注入初始状态；之后每次 submit 用 `Command(resume=...)` 恢复。
- 同步图与异步图使用**不同的 checkpoint 库文件**，避免 SQLite 锁竞争：
  - `data/checkpoints.sqlite`（CLI / Gradio，同步）
  - `data/checkpoints_api.sqlite`（SSE API，异步）

## API 协议

### POST `/api/consult/stream`

请求体：

```json
{
  "patient_id": "patient_001",
  "message": "最近一周爬楼后胸口发闷",
  "thread_id": "可选，不传则新建",
  "media_id": "可选，已上传的病历文档"
}
```

`thread_id` 语义：一次问诊 = 一个 thread。**续聊必须传回同一个 thread_id**，
后端据此判断是「新建会话」还是「恢复挂起点」。

`media_id` 语义：传了就先摄取这份病历再采集。首次问诊和续聊都能传——
续聊时传会走 `media_pending` 路径，摄取完再回到原来的采集阶段。

响应为 SSE（`text/event-stream`），每行 `data: {json}`。事件类型：

| type | 说明 | 字段 |
| --- | --- | --- |
| `node` | 某节点执行完成 | `node`, `reply`, `action`, `department` |
| `interrupt` | 图挂起，等待患者输入 | `interrupt_type`, `question`, `department` |
| `done` | 流程结束 | `status`, `department`, `phase`, `report_path`, `report_content`, `appointment`, `facts`, `missing_slots`, `triage`, `red_flags`, `urgent_flags`, `media`, `slot_sources`, `slot_conflicts`, `messages` |
| `error` | 执行出错 | `message` |

`interrupt_type` 取值：

- `ask_patient` — 需要患者回答问题（`question` 是问句）
- `confirm_appointment` — 需要患者确认是否挂号（`question` 是确认提示）

### POST `/api/media/upload`

上传病历文档（`multipart/form-data`，字段名 `file`），返回：

```json
{
  "media_id": "88dd3fc6e4f6",
  "filename": "出院小结.pdf",
  "size": 1819,
  "mime": "application/pdf",
  "message": "上传成功，请开始问诊，文档信息会自动预填。"
}
```

支持 `.pdf` / `.docx` / `.txt` / `.md`。这一步只落盘与校验，
真正的解析与抽取在问诊开始时由 `ingest_document` 节点完成（需要模型）。

### GET `/api/media/{media_id}`

查询某次上传的文件列表。

### POST `/api/media/audio`

上传录音并转写成文字（`multipart/form-data`，字段名 `file`），返回：

```json
{
  "media_id": "3f9a1c2b7d4e",
  "filename": "recording.wav",
  "text": "我这两天有点胸闷，活动后明显",
  "language": "Chinese",
  "duration": 4.2,
  "ok": true,
  "message": "识别完成，请确认文字无误后再发送。"
}
```

**转写文本不会自动进入问诊**。前端应把 `text` 回填到输入框，
让患者确认或修改后再提交——这是医疗场景的必要环节。

支持 `.wav` / `.mp3` / `.m4a` / `.flac` / `.ogg` / `.webm` / `.aac` / `.mp4`。
非 WAV 格式需要 librosa + soundfile；浏览器录音（webm/opus）还需要 ffmpeg。

模型加载较慢（首次十几秒），因此接口同步返回完整结果。

### GET `/api/consult/state?thread_id=xxx`

返回该 thread 的当前状态，用于前端刷新后恢复界面。

### GET `/api/consult/report?case_id=xxx`

返回 Markdown 报告全文（`text/markdown`）。

## 前端接入要点

1. 首次进入页面时生成一个 `thread_id`（如 uuid），存到 localStorage。
2. `POST /api/consult/stream` 带上 `patient_id` + `thread_id` + 首句。
3. 解析 SSE：
   - `node` → 把 `reply` 追加到聊天记录（按 `node` 区分样式：
     `collect` 问诊助手 / `specialty_collect` 科室预问诊 / `triage` 导诊助手）
   - `interrupt` → 显示 `question`，等待用户输入
   - `done` → 展示报告 Markdown（`report_content`）与挂号信息
4. 用户回答后，**用同一个 `thread_id`** 再次 POST，`message` 填用户回答。
   后端会自动用 `Command(resume=...)` 恢复挂起点。
5. 刷新页面 → `GET /api/consult/state?thread_id=xxx` 恢复。

## 测试

```powershell
# CLI 端到端
python -m patient_agent.graph.smoke --case test_cardiology_001

# SSE 接口端到端
python -m patient_agent.graph.api_smoke --case test_cardiology_001

# Gradio 界面逻辑端到端
python -m patient_agent.graph.gradio_smoke --case test_cardiology_001

# 病历文档摄取（不依赖模型）
python -m patient_agent.graph.media.smoke

# 带文档 vs 不带文档的采集轮次对比
python -m patient_agent.graph.media.compare
```

可选 case：`test_cardiology_001` / `test_respiratory_001` /
`test_gastroenterology_001` / `test_dermatology_001` / `test_neurology_001` /
`test_urgent_001` / `test_handoff_001`

## 实现要点与坑

1. **interrupt 节点必须无副作用**。LangGraph 恢复时会从头重跑该节点，
   把写文件、扣款之类副作用放进去会执行两次。
   本项目把副作用拆到了 `booking_node`（在 interrupt 之后）。

2. **恢复必须用 `Command(resume=...)`**。用 `update_state` + `invoke(None)`
   会让 `ask_patient` 再次触发 `interrupt`，形成死循环。

3. **同步图与异步图要分开**。`SqliteSaver` 不支持 async 方法，
   `graph.astream()` 必须配 `AsyncSqliteSaver`，否则运行时报
   "The SqliteSaver does not support async methods"。

4. **`AsyncSqliteSaver` 的连接必须保持存活**。用 `from_conn_string` 时
   `async with` 退出即关闭连接，之后调用会报 "no active connection"。
   `server.py` 用 lifespan + `AsyncExitStack` 持有。

5. **SSE 事件类型不要被 payload 覆盖**。`{"type": "interrupt", **payload}`
   中 payload 自带 `type` 字段会把外层覆盖掉，前端就识别不出中断事件。
   所以单独抽了 `_interrupt_event()`，用 `interrupt_type` 承载原始类型。

6. **节点函数的额外参数要用 `partial` 绑定**。`collect_node(state, llm)`
   里的 `llm` 不是 LangGraph 注入的参数，直接注册会报缺参数。

7. **`thread_id` 与 `patient_id` 是两回事**。前者是「一次问诊会话」，
   后者是「患者档案」。患者长期记忆仍按 `patient_id` 累积到
   `data/patients/{patient_id}.json`。

8. **急诊判定用代码兜底，不靠模型**。`core/departments.py` 里每个科室有
   `RedFlagRule` 正则 + 否定词处理（「没有持续胸痛」不算命中「持续胸痛」）。
   命中 urgent 规则时**强制** `urgency=urgent`，可以推翻模型的乐观结论。

9. **收束有确定性兜底**。`collect_node` 在模型既不给问题又想结束时
   直接置 `collection_done`，避免空提问死循环；轮数上限由
   `MAX_GENERAL_TURNS` / `MAX_SPECIALTY_TURNS` 控制。

10. **还没做的**：token 级流式输出。当前 `GraphLLM.ask()` 是一次性返回，
    前端拿到的是整段回复。要逐字输出需加 `ask_stream()`，
    走 `client.chat.completions.stream()`。

## 病历文档提取（media 模块）

### 为什么不能把文档塞进 messages

采集智能体问什么、什么时候停，只看 `slots`，不看对话：

```python
missing = [slot for slot in spec.required_slots if slot not in filled_before]
```

而且拼进 prompt 的对话会被 `recent_dialogue` 截断到每条 400 字。
一份 5000 字的出院小结塞进 `messages`，模型只看到前 400 字。

所以文档要**直接写 `slots`**，采集自然就不问了。

### 流程

```text
上传 .pdf/.docx
  → media/documents.py 解析文本 + 章节切分（纯 Python，不调模型）
  → media/extract.py   一次 LLM 调用抽成 slot_patch
  → media/ingest.py    合并进 state["slots"]，记 slot_sources / slot_conflicts
  → collect 的 missing 自动变短 → 只问缺口
```

### 关键语义：既往病历 ≠ 本次主诉

一份上次就诊的出院小结如果被抽成本次的 `主诉`/`病程`，
分诊和最终报告描述的就是另一个时间点的病情。

`media/extract.py` 的处理：

| 文档里的字段 | 去处 |
| --- | --- |
| 既往史 / 用药史 / 过敏史 / 手术史 / 个人史 / 家族史 / 婚育史 | `slots` |
| 体格检查、辅助检查、化验数值 | `facts["既往资料"]` |
| 主诉 / 现病史 / 病程 | `facts["既往资料"]["既往主诉"]`（**不填本次槽位**） |

只有 `doc_type == "outpatient_record"` 且日期在 90 天内，才允许写入本次主诉。

### 合并规则（写死在 `media/ingest.py`）

1. **患者口述优先**：`slot_sources[slot] == "patient"` 时文档值不覆盖，只记冲突；
2. **文档内部冲突**：多份文档同一槽位不一致 → 取最新日期那份，并留痕；
3. **冲突不静默**：`slot_conflicts` 会拼进 `collect` 的 prompt，
   让模型下一轮顺带问一句「病历写 X，您上次说没有，以哪个为准」。

### 降级

| 情况 | 行为 |
| --- | --- |
| 加密 PDF / `.doc` / `.wps` | 拒绝，提示另存为无密码 PDF 或 .docx |
| 扫描件（每页平均 < 50 字） | 标记 `needs_ocr`，提示识别质量有限，主流程继续 |
| 超页数 / 超字数 | 拒绝，提示拆分 |
| 抽取失败 / JSON 不合法 | `status="documents_failed"`，主流程继续 |

### 验收数据

用 `python -m patient_agent.graph.media.compare` 跑同一份出院小结：

```text
预填槽位（3 项）：['既往史', '用药史', '过敏史']
问到既往史/用药/过敏类问题：0 -> 0
  ✓ 文档已有的既往史/用药/过敏信息没有重复提问
```

`collection_turns` 这个指标本身抖动较大——固定回答会把模型引向不同分支，
有时反而多问几个症状问题。真正稳定的指标是**预填槽位数**和
**重复提问率**：文档覆盖的槽位不再进入 `missing`，采集自然不问。

## 语音识别（media/asr.py）

### 为什么不复用 GraphLLM

ASR 是本地多模态模型，跟走 OpenAI 兼容接口的对话模型是两套东西：
- 权重 4.7GB bf16，**离线加载**在 `patient_agent/models/Qwen3-ASR-1.7B`
- 输入是音频不是文本
- 输出是纯文本，没有 JSON 结构要求

所以单独一个 `media/asr.py`，只用 `GraphLLM` 之外的通道。

### 流程

```text
录音文件
  → _read_audio      读字节 + 格式校验
  → _decode_to_wav   统一转 16kHz 单声道 16bit PCM
  → load_model       懒加载 + 全局单例（4.7GB 不能反复加载）
  → model.transcribe((ndarray, sr), language=...)
  → Transcript(text, language, duration)
```

### 关键约束

1. **懒加载 + 单例**。权重加载十几秒，必须复用。
   `_MODEL` / `_MODEL_ERROR` 是模块级全局，用 `_MODEL_LOCK` 保护。

2. **显存自适应**。RTX 4060 Laptop 8GB，系统本身占 3GB，只剩 4~5GB，
   而模型 bf16 就占 4.7GB。`_resolve_device()` 估算权重大小，
   显存不够就开 `max_memory` + `offload_folder` 做 CPU offload。

3. **音频必须 16kHz 单声道**。特征提取器是 `WhisperFeatureExtractor`
   （`preprocessor_config.json`: `sampling_rate` 16000, `feature_size` 128）。
   WAV 走纯 Python 重采样路径（不依赖 librosa），其他格式用 librosa。

4. **转写不自动进对话**。Gradio 把文本回填输入框让患者核对；
   API 返回文本让前端回填。医疗场景不能让未经确认的识别结果直接进问诊。

5. **失败不抛异常**。`transcribe()` 永远返回 `Transcript`，
   `error` 字段带用户能看懂的提示。语音是可选输入通道，
   识别失败不该打断问诊。
