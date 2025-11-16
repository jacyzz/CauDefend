## Causal Code Defender (CCD)

统一的代码领域后门防御框架，面向训练与推理阶段的鲁棒性评估与防御。框架包含：
- IST（语义等价代码变换）：基于 tree-sitter 的代码风格/结构转换、评估与训练数据构建
- Inference（推理与等义评测）：使用多样化 beam search 生成候选代码，评估等义性
- Training（防御模型训练）：Tuna 与 PRO（含 LoRA）训练脚手架
- 统一 CLI、配置与稳定 JSONL 输出，便于后续前端接入

### 快速开始
1) 安装依赖
```
pip install -r requirements.txt
```

2) 查看 CLI
```
python -m ccd.cli --help
python -m ccd.cli ist --help
python -m ccd.cli dataset --help
python -m ccd.cli infer --help
python -m ccd.cli train --help
```

3) 使用 IST 构建训练数据（示例）
```
python -m ccd.cli dataset make \
  --input data/sample.jsonl \
  --input-field code \
  --language python \
  --style comment_header \
  --strategy simple \
  --output data/ist_dataset.jsonl
```

4) 推理与等义性评测（示例）
```
python -m ccd.cli infer eval \
  --model codellama/CodeLlama-7b-hf \
  --dataset data/humaneval_like.jsonl \
  --language python \
  --output outputs/infer_results.jsonl \
  --num-beams 4 --num-return-sequences 4 --diversity-penalty 0.1
```

### 目录结构
```
CauDefend/
  ├─ README.md
  ├─ requirements.txt
  ├─ configs/
  │   ├─ ist.yaml
  │   └─ infer.yaml
  ├─ scripts/
  │   ├─ ist/
  │   │   └─ eval_humaneval.sh
  │   └─ dataset/
  │       └─ make_ist.sh
  └─ src/
      └─ ccd/
          ├─ __init__.py
          ├─ cli.py
          ├─ config.py
          ├─ utils/
          │   └─ logging.py
          ├─ ist/
          │   ├─ languages.py
          │   ├─ styles.py
          │   ├─ transfer.py
          │   └─ transform/
          ├─ datasets/
          │   └─ make_ist_dataset.py
          ├─ inference/
          │   └─ run_defense_eval.py
          └─ training/
              ├─ tuna/
              │   └─ train.py
              └─ pro/
                  └─ train.py
```

### 设计要点
- 统一 CLI（Typer），统一 JSONL 输出 schema，便于前端消费
- 配置文件（YAML）+ 命令行参数，可控且简洁
- Inference 使用多样化 beam search（num_beams=num_return_sequences，num_beam_groups>1，diversity_penalty>0）
- IST 先支持 Python，逐步扩展到多语言（tree-sitter）

### 后续
- 集成 OpenBackdoor 训练/评估脚本（参考 Frontdoor-Adjustment-Backdoor-Elimination）
- 完整的人类评测集（HumanEval/MBPP）测试器接入
- 训练脚本完善（Tuna/PRO 全量实现、日志与指标追踪）

### 仅迁移 IST（transfer + 风格）指引
1) 将你在 `CausalCodeDefense/src/IST/transform/` 下的实现（包含 `config.py`, `lang.py`, 各语言变换等）原样复制到：
```
src/ccd/ist/transform/
```
2) 如需扩展/修改风格码映射，编辑：
```
src/ccd/ist/styles.py
```
3) 调用示例（纯后端）：
```python
from ccd.ist import StyleTransfer
st = StyleTransfer(language="python")
code_after, ok = st.transfer(styles=["-1.1","0.5"], code="...your code...")
```
4) 安装依赖：
```
pip install -r requirements.txt
```


