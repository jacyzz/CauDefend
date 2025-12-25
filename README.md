## CCD（Causal Code Defender）— 后端/前端启动与参数说明

### 项目概览
- 后端：FastAPI（`src/ccd/server/main.py`），提供 IST 与 Inference 的 REST API
- 推理引擎：Hugging Face Transformers（`src/ccd/inference/engine.py`），支持普通 beam-search、多样化 beam-search、PEFT/LoRA
- 前端：React + Vite + TypeScript + Ant Design（`ccd-frontend/`），包含 IST、Inference 页面

目录要点：
```
CauDefend/
  ├─ README.md
  ├─ requirements.txt
  ├─ scripts/server/run_api.sh
  ├─ src/ccd/server/main.py         # FastAPI 入口（/api/ist/*, /api/infer/*）
  ├─ src/ccd/inference/engine.py    # 模型加载与生成逻辑
  └─ ccd-frontend/                  # 前端
      ├─ vite.config.ts             # 代理 /api -> 127.0.0.1:8001
      └─ package.json               # 前端脚本
```

---

### 环境准备（后端）
- Python 3.10+（推荐 Conda）
- 安装依赖：

```bash
cd /home/nfs/u2023-zlb/CauDefend
pip install -r requirements.txt
# 注意：torch 需按你机器的 CUDA 版本安装（参见 https://pytorch.org/get-started/locally/）
# 例如：pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

可选：设置 `PYTHONPATH` 方便本地运行
```bash
export PYTHONPATH=/home/nfs/u2023-zlb/CauDefend/src:$PYTHONPATH
```

---

### 启动后端（FastAPI）
- 脚本方式（默认使用 `--reload`，便于开发）：

```bash
bash scripts/server/run_api.sh 8001
```

如需使用远端模型提供商（OpenAI/DeepSeek/中转站等 OpenAI-compatible 接口），请先按文档配置后端环境变量（示例文件：`.env.example`）：
- 说明文档：[docs/remote-provider.md](docs/remote-provider.md)

- 或手动启动：

```bash
export PYTHONPATH=/home/nfs/u2023-zlb/CauDefend/src:$PYTHONPATH
python -m uvicorn ccd.server.main:app --host 0.0.0.0 --port 8001 --reload
```

健康检查：
```bash
curl http://127.0.0.1:8001/api/health
```

---

### 环境准备与启动（前端）
- Node.js ≥ 18
- 安装依赖并启动开发服务：

```bash
cd /home/nfs/u2023-zlb/CauDefend/ccd-frontend
npm install    # 或 pnpm install / yarn install
npm run dev
```

默认访问：
- 前端：http://127.0.0.1:5173
- 代理配置：`ccd-frontend/vite.config.ts` 中将 `'/api'` 代理到 `http://127.0.0.1:8001`，请确保后端监听端口为 `8001`。如需修改后端端口，请同步调整该文件的 `target`。

---

### 关键 API 与示例

IST：
- `GET /api/ist/styles`：列出可用风格
- `POST /api/ist/transform_text`：单段代码转换
- `POST /api/ist/transform_dataset`：数据集批量转换（支持固定/随机风格池、转换前后预览）

Inference：
- `POST /api/infer/generate`：单条输入生成候选
- `POST /api/infer/dataset`：数据集推理（读入 `input_path`，写出到 `output_path`）
- `POST /api/infer/unload`：按模型配置释放已缓存模型与显存
- `POST /api/infer/unload_all`：释放全部模型与显存

示例 1：单条推理
```bash
curl -X POST http://127.0.0.1:8001/api/infer/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "def add(a,b):\n    return a+b",
    "model": {
      "model": "/path/to/model",
      "dtype": "float16",
      "device_map": "auto",
      "trust_remote_code": false,
      "low_cpu_mem_usage": false,
      "use_safetensors": true
    },
    "prompt": {
      "template_yaml": null,
      "system_prompt_text": "You are a helpful assistant."
    },
    "gen": {
      "max_new_tokens": 128,
      "do_sample": false,
      "num_beams": 4,
      "num_return_sequences": 4,
      "num_beam_groups": 1,
      "diversity_penalty": 0.0,
      "temperature": 1.0,
      "top_p": 1.0,
      "seed": 123456
    },
    "return_decoded": false,
    "unload_after": false
  }'
```

示例 2：数据集推理（多候选多行输出）
```bash
curl -X POST http://127.0.0.1:8001/api/infer/dataset \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/path/to/input.jsonl",
    "output_path": "/path/to/output.jsonl",
    "field": "canonical_solution",
    "output_field": "generation",
    "model": {
      "model": "/path/to/model",
      "dtype": "float16",
      "device_map": "auto",
      "trust_remote_code": false,
      "low_cpu_mem_usage": false,
      "use_safetensors": true
    },
    "prompt": {
      "template_yaml": null,
      "system_prompt_text": "You are a helpful assistant."
    },
    "gen": {
      "max_new_tokens": 256,
      "do_sample": false,
      "num_beams": 4,
      "num_return_sequences": 2,
      "num_beam_groups": 1,
      "diversity_penalty": 0.0
    },
    "emit_flat": true,
    "write_mode": "generation",
    "limit": 0,
    "unload_after": false
  }'
```

多候选多行的规则：
- 当 `emit_flat=true` 或 `num_return_sequences>1` 时，会为同一原始样本生成多行，每行一个候选，其他字段保持一致。
- `write_mode="generation"` 时写到 `output_field`（默认 `"generation"`）；`write_mode="overwrite"` 时覆盖原 `field`。
- 始终写入 `output_path`，不会原地修改输入数据集。

---

### 常用参数解释（后端）
- **模型加载（model）**
  - `model`: 模型目录或 HF 名称；支持合并后的权重（含 sharded safetensors）
  - `dtype`: `"float16" | "bfloat16" | "auto"`，建议 `float16`（更广泛兼容）
  - `device_map`: `"auto" | "cuda:0" | "cpu"` 等；`"auto"` 自动放置到可用 GPU/CPU
  - `trust_remote_code`: 一些社区模型需要启用自定义代码
  - `low_cpu_mem_usage`: 降低加载时的 CPU 内存占用
  - `use_safetensors`: 优先/强制使用 safetensors；若目录含 safetensors 会自动启用
  - `base_model`/`peft_adapter`/`peft_merge`: 支持加载 LoRA 适配器并可选择 merge

- **提示词（prompt）**
  - `template_yaml`: 兼容 LLaMA-Factory 风格的模板 YAML，可将 `system_prompt` 填充进模板
  - `system_prompt_text`: 直接给定 system prompt 文本

- **生成参数（gen）**
  - `max_new_tokens`: 最大新生成 token 数
  - `do_sample`: 采样开关（关闭时为贪心/beam）
  - `temperature` / `top_p`: 采样时控制多样性
  - `num_beams`: beam 数
  - `num_return_sequences`: 返回候选数
  - `num_beam_groups` / `diversity_penalty`: 分组多样化 beam-search；若 `num_beam_groups<=1` 会自动把 `diversity_penalty` 设为 0 以避免报错
  - `seed`: 随机种子（用于采样/可重复性）

- **数据集推理**
  - `field`: 输入数据中要处理的字段（如 `"canonical_solution"`）
  - `write_mode`: `"generation"`（写到 `output_field`）或 `"overwrite"`（覆盖原字段）
  - `output_field`: `write_mode="generation"` 时生效
  - `emit_flat`: 多候选是否展开为多行（推荐 `true`）
  - `limit`: 仅处理前 N 行（0 表示不限制）
  - `progress`: 服务器端控制台显示 tqdm 进度
  - `unload_after`: 本次请求后释放模型与显存（减少 VRAM 长驻）

- **GPU 内存管理**
  - 服务端做了模型缓存以避免重复加载；若需要释放显存：
    - 请求中传 `unload_after: true`
    - 或调用 `POST /api/infer/unload`、`POST /api/infer/unload_all`

---

### 常见问题
- **多样化 beam 的 dtype 报错（bfloat16）**  
  有些模型在 `bfloat16 + group beam search` 时会报 dtype mismatch。引擎会自动回退为取消分组或临时 cast 到 `float16` 再试；若仍异常，建议手动设置：`num_beam_groups=1` 或 `dtype=float16`。

- **diversity_penalty 报错**  
  当 `num_beam_groups=1` 时设置了 `diversity_penalty>0` 会触发 HF 检查。我们已在服务端自动将其归零以保证普通 beam 正常执行。

- **前端 502/ECONNREFUSED**  
  确保后端在 `8001` 端口，且 `ccd-frontend/vite.config.ts` 的代理 `target` 与之对应。

- **显存未释放/多次推理显存占用累积**  
  使用 `unload_after: true`，或调用 `/api/infer/unload(_all)` 主动释放。

- **换行显示不一致（`\n` vs 实际换行）**  
  有的候选包含字面量 `\n`，有的包含真实换行。写回 JSONL 时会按字符串原样写入，不会出错；若希望统一，可在后端写回前进行规范化（例如将字面量 `\\n` 统一替换为真实换行，或反之）。

---

### 前端功能概览
- IST：单条转换、数据集转换（固定/随机风格池），右侧日志与预览；支持查看数据集字段与单条记录前后对比
- Inference：单条与数据集推理，参数面板分组（快速参数、模型/LoRA、提示模板、高级生成），支持 beam-search、多候选展开，数据集推理含进度条

如需自定义参数布局或新增选项，可编辑：
- `ccd-frontend/src/pages/infer/Single.tsx`
- `ccd-frontend/src/pages/infer/Dataset.tsx`
- `ccd-frontend/src/pages/ist/*`
