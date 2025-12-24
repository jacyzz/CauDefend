# 远端模型提供商（OpenAI 兼容）配置与使用

本项目支持两类推理后端：
- `provider=local`：本地 Hugging Face 模型（现有逻辑）。
- `provider!=local`：后端代理调用远端 LLM（OpenAI-compatible `chat/completions`）。

远端推理的 API Key **只配置在后端环境变量**，前端不会持有/传递 key。

---

## 1. 什么是 OpenAI-compatible

很多模型提供商（或中转站）会提供与 OpenAI 风格一致的接口：
- 路径：通常是 `POST /v1/chat/completions`
- 鉴权：通常 `Authorization: Bearer <API_KEY>`
- 请求体：`{"model": "...", "messages": [{"role":"user","content":"..."}], ...}`
- 返回：`choices[].message.content`

**兼容的意义**：你只要实现一套客户端，就能切换不同提供商（只换 `base_url` 和 `model` 名称）。

---

## 2. 环境变量（后端配置）

项目提供示例文件：.env.example

你可以把它复制成 `.env` 后，在启动后端前导入：

```bash
cd /home/wood/CauDefend
cp .env.example .env
# 编辑 .env，填入真实 key
set -a
source .env
set +a
bash scripts/server/run_api.sh 8001
```

备注：后端会在启动/首次读取配置时**自动尝试加载**项目目录（向上搜索）的 `.env` 文件，所以一般不必手动 `source .env`；生产环境仍建议用真正的环境变量/密钥管理。

### 2.1 provider 与环境变量前缀规则

后端读取环境变量的规则是：

- 当 `provider` 为 `openai_compatible` 或 `remote`：
  - 读取 `CCD_REMOTE_BASE_URL`
  - 读取 `CCD_REMOTE_API_KEY`
  - 可选：`CCD_REMOTE_CHAT_PATH`（默认 `/v1/chat/completions`）

- 当 `provider` 为任意其他字符串（如 `openai` / `deepseek` / `xxx`）：
  - 读取 `CCD_<PROVIDER_UPPER>_BASE_URL`
  - 读取 `CCD_<PROVIDER_UPPER>_API_KEY`
  - 可选：`CCD_<PROVIDER_UPPER>_CHAT_PATH`

示例：
- `provider=deepseek` → `CCD_DEEPSEEK_BASE_URL` + `CCD_DEEPSEEK_API_KEY`
- `provider=openai` → `CCD_OPENAI_BASE_URL` + `CCD_OPENAI_API_KEY`

### 2.2 超时与重试（可选）

- `CCD_REMOTE_TIMEOUT_S`：默认 60
- `CCD_REMOTE_MAX_RETRIES`：默认 3

对 429/5xx 会做简单指数退避重试。

## 2.3 关于 base_url / chat_path 的写法（避免重复 /v1）

很多中转站会给两种等价写法，你任选其一即可：

- 写法 A（推荐）：
  - `CCD_REMOTE_BASE_URL=https://poloai.top`
  - `CCD_REMOTE_CHAT_PATH=/v1/chat/completions`

- 写法 B（也支持）：
  - `CCD_REMOTE_BASE_URL=https://poloai.top/v1/`
  - `CCD_REMOTE_CHAT_PATH=chat/completions`

后端会做一次简单的去重，避免拼成 `.../v1/v1/...`。

---

## 3. API 使用方式

### 3.1 单条推理（/api/infer/generate）

当你用远端 provider 时：
- `model.model` 填远端的模型名（例如 `gpt-4o-mini`、`deepseek-chat`）
- 本地 HF 专用字段（dtype/device_map/LoRA）会被忽略

示例（remote）：

```bash
curl -X POST http://127.0.0.1:8001/api/infer/generate \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai_compatible",
    "input_text": "def add(a,b):\n    return a+b",
    "model": { "model": "gpt-4o-mini" },
    "prompt": { "system_prompt_text": "You are a helpful assistant." },
    "gen": { "max_new_tokens": 128, "do_sample": false, "temperature": 1.0, "top_p": 1.0, "num_return_sequences": 2 }
  }'
```

备注：远端不一定支持 beam/group beam；这些参数可能会被提供商忽略。

补充：`/v1/chat/completions` 一般**只支持 POST**，用 `curl -i <url>`（GET）测试会得到 404，这是正常的。

### 3.2 数据集推理（结构化异步）

结构化 dataset（前端进度条）继续走：
- `POST /api/infer/dataset_structured_async`
- `GET /api/infer/progress?task_id=...`

你只需在请求体里加：

```json
{ "provider": "openai_compatible", "model": { "model": "..." }, "...": "..." }
```

输出格式与原逻辑保持一致：
- `structured_variants`：写入 `output` 数组（variants 列表）
- `single_field`：写入指定 field（可能为候选列表）

---

## 4. 前端怎么用

在 Inference 页面：
- 选择 `推理后端 (provider)` 为 `local` / `openai_compatible` / `openai` / `deepseek`
- `模型路径/名称`：
  - local：填本地模型目录
  - remote：填远端模型名

---

## 5. 安全建议

- 不要把 key 写入前端或提交到 git。
- 生产环境建议只在服务端环境变量或密钥管理服务中注入 key。
- 若使用中转站，确保 `base_url` 是可信域名并启用 HTTPS。
