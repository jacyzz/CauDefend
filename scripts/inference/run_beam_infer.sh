#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
# Minimal HF beam-search inference (no vLLM), FABE-style prompt
# Example:
#   bash scripts/inference/run_beam_infer.sh \
#     --model codellama/CodeLlama-7b-hf \
#     --input /path/to/humaneval_java.jsonl \
#     --field canonical_solution \
#     --output /path/to/out.jsonl \
#     --template-yaml /home/nfs/u2023-zlb/CauDefend/templates/code_security_cleanup.yaml \
#     --system-prompt-text "你是资深代码安全专家..." \
#     --num-beams 2 --num-return-sequences 2 --emit-flat \
#     --low-cpu-mem-usage --use-safetensors

MODEL="/home/nfs/share-yjy/dachuang2025/defense_model/dscoder-6.7b-pro-merged3"
INPUT="/home/nfs/share-yjy/dachuang2025/codefuse-evaluation/codefuseEval_202503/data/code_completion/IST_eval/humaneval_java.jsonl"
FIELD="canonical_solution"
OUTPUT="/home/nfs/share-yjy/dachuang2025/codefuse-evaluation/codefuseEval_202503/data/code_completion/model_fix/humaneval_java.jsonl"
TEMPLATE_YAML="/home/nfs/u2023-zlb/CauDefend/templates/code_security_cleanup.yaml"
SYSTEM_PROMPT_TEXT="你是资深代码安全与重构专家。任务：在保持功能等价的前提下，去除/修复代码中的潜在后门，确保可直接替换回原字段。"
NUM_BEAMS=2
NUM_RETURN_SEQS=2
NUM_BEAM_GROUPS=1
DIVERSITY=0.0
MAX_NEW_TOKENS=1024
DTYPE=bfloat16
DEVICE_MAP=auto
DO_SAMPLE=1
TEMPERATURE=0.1
TOP_P=1.0
LIMIT=0
SEED=123456
EMIT_FLAT=1
LOW_CPU=0
USE_SAFE=0
WRITE_MODE="generation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --field) FIELD="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --template-yaml) TEMPLATE_YAML="$2"; shift 2 ;;
    --system-prompt-text) SYSTEM_PROMPT_TEXT="$2"; shift 2 ;;
    --num-beams) NUM_BEAMS="$2"; shift 2 ;;
    --num-return-sequences) NUM_RETURN_SEQS="$2"; shift 2 ;;
    --num-beam-groups) NUM_BEAM_GROUPS="$2"; shift 2 ;;
    --diversity-penalty) DIVERSITY="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --device-map) DEVICE_MAP="$2"; shift 2 ;;
    --do-sample) DO_SAMPLE=1; shift 1 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --emit-flat) EMIT_FLAT=1; shift 1 ;;
    --low-cpu-mem-usage) LOW_CPU=1; shift 1 ;;
    --use-safetensors) USE_SAFE=1; shift 1 ;;
    --write-mode) WRITE_MODE="$2"; shift 2 ;;
    *) echo "[ERR] Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$INPUT" || -z "$FIELD" || -z "$OUTPUT" ]]; then
  echo "[ERR] Missing required args: --model --input --field --output" >&2
  exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../src && pwd)"
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

CMD=(python -m ccd.inference.beam_infer
  --model "$MODEL"
  --input "$INPUT"
  --field "$FIELD"
  --output "$OUTPUT"
  --write-mode "$WRITE_MODE"
  --num-beams "$NUM_BEAMS"
  --num-return-sequences "$NUM_RETURN_SEQS"
  --num-beam-groups "$NUM_BEAM_GROUPS"
  --diversity-penalty "$DIVERSITY"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --dtype "$DTYPE"
  --device-map "$DEVICE_MAP"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --limit "$LIMIT"
  --seed "$SEED"
)
[[ -n "$TEMPLATE_YAML" ]] && CMD+=(--template-yaml "$TEMPLATE_YAML")
[[ -n "$SYSTEM_PROMPT_TEXT" ]] && CMD+=(--system-prompt-text "$SYSTEM_PROMPT_TEXT")
[[ "$DO_SAMPLE" == "1" ]] && CMD+=(--do-sample)
[[ "$EMIT_FLAT" == "1" ]] && CMD+=(--emit-flat)
[[ "$LOW_CPU" == "1" ]] && CMD+=(--low-cpu-mem-usage)
[[ "$USE_SAFE" == "1" ]] && CMD+=(--use-safetensors)

echo "[INFO] Running: ${CMD[*]}" >&2
"${CMD[@]}"


