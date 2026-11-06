# Patient Agent

极简线上挂号前预问诊 Agent。

核心流程是：

```text
患者输入 -> 主分诊 Agent -> 科室 Skill 子 Agent -> 可回退重分诊 -> 报告生成 -> 写入患者记忆 -> 可选一键挂号
```

## 两套实现

| 实现 | 入口 | 说明 |
| --- | --- | --- |
| `simple_agent` | `python -m patient_agent.simple_agent.app` | 纯命令行，手写 `while` 循环控制流程 |
| `graph` | `python -m patient_agent.graph.cli` | LangGraph 版本，双 Agent 架构（采集 + 分诊），附带 FastAPI + SSE 接口和 Gradio 界面 |

`graph` 版本是 `simple_agent` 的重构，两套并存、互不影响。详细说明见 `docs/graph_agent.md`。

## 配置

填写根目录 `.env`：

```bash
OPENAI_API_KEY=你的 API key
OPENAI_MODEL=deepseek-flash
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_MODE=chat_completions
PATIENT_AGENT_MAX_SPECIALTY_TURNS=4
PATIENT_AGENT_MAX_SWITCHES=2
```

如果使用官方 OpenAI，可以把 `OPENAI_BASE_URL` 留空，并把 `OPENAI_API_MODE` 设置为 `auto` 或 `responses`。代码也兼容 `OPENAI_MODEL_URL` 作为 `OPENAI_BASE_URL` 的别名。

> `.env` 已被 `.gitignore` 忽略，不要把真实 key 写进 README 或提交到仓库。

## 安装

```bash
pip install -r requirements.txt
```

## 运行

### 命令行版（simple_agent）

```bash
python -m patient_agent.simple_agent.app --patient-id patient_001
```

### 命令行版（LangGraph）

```bash
python -m patient_agent.graph.cli --patient-id patient_001

# 直接给首句，非交互
python -m patient_agent.graph.cli --patient-id patient_001 --input "最近一周爬楼后胸口发闷"

# 带病历文档（会先解析并预填既往史/用药史/过敏史，减少重复提问）
python -m patient_agent.graph.cli --patient-id patient_001 --document "data/samples/出院小结.pdf"
```

### API 版（LangGraph + FastAPI + SSE）

```bash
python -m patient_agent.graph.server
# 或
uvicorn patient_agent.graph.server:app --reload --port 8000
```

接口文档：<http://127.0.0.1:8000/docs>

### Gradio 界面版

```bash
# 独立运行（7860 端口）
python -m patient_agent.graph.gradio_app

# 挂到 FastAPI 上，与 API 共用 8000 端口
python -m patient_agent.graph.server --with-gradio
```

界面地址：<http://127.0.0.1:8000/>

### 病历文档提取（减少重复提问）

上传 PDF / DOCX 病历，系统会解析文本 → 模型抽取结构化槽位 → 直接预填
`state["slots"]` → 采集智能体自动跳过这些槽位，只问缺口。

```bash
# 生成一份样例病历 PDF
python -m patient_agent.graph.media.make_sample

# 解析/抽取/合并规则的冒烟测试（不依赖模型）
python -m patient_agent.graph.media.smoke

# 对比带文档 vs 不带文档的采集轮次与重复提问率
python -m patient_agent.graph.media.compare
```

接口：

```text
POST /api/media/upload   上传病历文档，返回 media_id
GET  /api/media/{id}     查询上传信息
```

开始问诊时把 `media_id` 一起传给 `POST /api/consult/stream`，
图里的 `ingest_document` 节点会自动完成解析与预填。

关键设计：

- **既往病历 ≠ 本次主诉**。出院小结 / 检验报告里的主诉、病程一律
  写进 `facts["既往资料"]`，不填本次 `主诉`/`病程`，避免误导分诊。
- **患者口述优先**。`slot_sources[slot] == "patient"` 时文档值不覆盖，
  只在 `slot_conflicts` 留痕，下一轮顺带向患者确认。
- **降级不崩**。加密 PDF、`.doc`、扫描件、超限文件都会被明确拒绝或提示，
  主流程继续。

Gradio 界面在「上传病历文档（可选）」折叠面板里直接上传。

### 语音识别（Qwen3-ASR）

患者可以用语音和问诊助手交流。录音后由本地 `Qwen3-ASR-1.7B` 模型转写，
**转写文本回填到输入框，患者核对/修改后再发送**——医疗场景里
「胸闷」识别成「胸焖」这类错误必须留有人工纠正的机会。

模型位置：`patient_agent/models/Qwen3-ASR-1.7B`（离线加载，不联网）。

安装依赖（注意 `qwen-asr` 会锁定 `transformers==4.57.6`）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install qwen-asr librosa soundfile
```

⚠️ `qwen-asr` 会把 `gradio` 降到 `6.17.3`，并要求 `transformers==4.57.6`。
在只跑本项目的环境里没有影响；若同环境还要用别的 transformers 版本模型，需要单独隔离。

实测环境：RTX 4060 Laptop 8GB，`device=cuda:0` / `offload=False`，
显存占用 3.9GB，7.9 秒音频转写耗时 2.5 秒（模型复用后）。

配置（`.env`，都有默认值）：

```bash
PATIENT_AGENT_ASR_MODEL_PATH=patient_agent/models/Qwen3-ASR-1.7B
PATIENT_AGENT_ASR_DEVICE=auto      # auto | cuda | cpu
PATIENT_AGENT_ASR_DTYPE=bfloat16   # bfloat16 | float16 | float32
PATIENT_AGENT_ASR_LANGUAGE=Chinese # 留空则自动检测
PATIENT_AGENT_ASR_MAX_SECONDS=120
```

用法：

```text
接口    POST /api/media/audio   上传录音，返回 {text, language, duration}
Gradio  麦克风按钮 →「识别语音」→ 文本回填输入框
CLI     python -m patient_agent.graph.cli --patient-id p1 --audio rec.wav
```

关键设计：

- **模型懒加载 + 全局单例**。4.7GB 权重加载要十几秒，只加载一次。
- **显存自适应**。`auto` 模式下估算权重占用，显存不够（RTX 4060 8GB
  通常只剩 4~5GB）自动开 CPU offload，避免 OOM。
- **音频统一预处理**。浏览器录音常见 webm/opus，全部转成 16kHz 单声道
  16bit PCM WAV 再送模型。WAV 走纯 Python 路径，不依赖 librosa。
- **降级不崩**。模型未装、格式不支持、录音太短、超时长上限，
  都返回明确的错误提示让患者改用文字输入，不中断问诊。

### 医学检查图像识别

尚未实现。后续会在 `media/` 下加 `vision.py`，复用同一套落盘与合并链路。

## 输出

流程结束时，医生预问诊报告只会保存到 `data/reports/{case_id}.md`，不会直接打印在命令行里。随后系统会询问是否需要一键挂号，输入 `y`、`yes` 或 `是` 会模拟挂号成功。

## 测试

```bash
# CLI 端到端
python -m patient_agent.graph.smoke --case test_cardiology_001

# SSE 接口端到端
python -m patient_agent.graph.api_smoke --case test_cardiology_001

# Gradio 界面逻辑端到端
python -m patient_agent.graph.gradio_smoke --case test_cardiology_001

# 病历文档摄取（不依赖模型）
python -m patient_agent.graph.media.smoke

# 语音识别模块（不依赖模型；--with-model 才会真正调用）
python -m patient_agent.graph.media.asr_smoke
python -m patient_agent.graph.media.asr_smoke --with-model samples/xxx.wav
```

可选 case：`test_cardiology_001` / `test_respiratory_001` / `test_gastroenterology_001` / `test_dermatology_001` / `test_neurology_001` / `test_urgent_001` / `test_handoff_001`

## 说明

项目逻辑见：

```text
docs/simple_agent_logic.md   # simple_agent 版逻辑
docs/graph_agent.md          # LangGraph 版逻辑与 API 协议
```

测试 case 见：

```text
tests/cases/README.md
```
