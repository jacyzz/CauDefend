#!/usr/bin/env bash
# Clean HumanEval poisoned datasets (C++/Java/Python) with a LoRA model and run evaluation.
# Usage: adjust the PARAMS section, then:
#   bash scripts/run_clean_and_eval.sh
#
# Notes:
# - Fill GEN_ARGS with your generation params (num_beams, num_return_sequences, etc.).
# - This script picks the latest epoch under each LoRA root (by name sort).
# - Requires: python env that can run beam_structured_infer.py and eval_*.py; jq (or use the Python fallback).
set -euo pipefail

#########################
# PARAMS (edit these)  #
#########################

# Poisoned input files
POISON_DIR="/home/nfs/share-yjy/dachuang2025/users/u2023-ckh/human-eval/data/poisoned"
POISON_CPP="${POISON_DIR}/humaneval_cpp_poisoned.jsonl"
POISON_JAVA="${POISON_DIR}/humaneval_java_poisoned.jsonl"
POISON_PY="${POISON_DIR}/humaneval_python_poisoned.jsonl"
# POISON_CPP="/home/nfs/u2023-zlb/CauDefend/data/humaneval_cpp_poisoned_test.jsonl"
# POISON_JAVA="/home/nfs/u2023-zlb/CauDefend/data/humaneval_java_poisoned_test.jsonl"
# POISON_PY="/home/nfs/u2023-zlb/CauDefend/data/huamaneval_python_poisoned_test.jsonl"

# Output root
OUT_ROOT="/home/nfs/share-yjy/dachuang2025/users/u2023-zlb/data/HumanEval/clean_eval_runs"

# Models to test (base + lora root). If指向具体 epoch，latest_epoch_dir 会返回该路径自身。
MODEL_BASE_Q25="/home/nfs/share-yjy/dachuang2025/01_Pretrained_Models/Qwen/Qwen2.5-Coder-7B-Instruct"
LORA_ROOT_Q25="/home/nfs/share-yjy/dachuang2025/03_Model_Checkpoints/defense_finetuned/qwen2.5_7b_pro2_vsp/"

MODEL_BASE_Q3="/home/nfs/share-yjy/dachuang2025/01_Pretrained_Models/Qwen/Qwen3-4B-Instruct-2507"
LORA_ROOT_Q3="/home/nfs/share-yjy/dachuang2025/03_Model_Checkpoints/defense_finetuned/Qwen3-4B_pro/"

# Generation params (FILL ME)
# Example: GEN_ARGS="--num-beams 4 --num-return-sequences 4 --max-new-tokens 512 --temperature 0.7 --top-p 0.9 --do-sample"
GEN_ARGS="--num-beams 4 --num-return-sequences 4 --max-new-tokens 512 --temperature 0.2 --top-p 0.9 --do-sample"

# Device/dtype (optional)
DEVICE_MAP="cuda:1"
DTYPE="float16"

# Prompt (choose one)
# SYSTEM_PROMPT_TEXT will be passed via --system-prompt-text
SYSTEM_PROMPT_TEXT_FILE=/home/nfs/u2023-zlb/CauDefend/scripts/prompt.txt   # e.g., /path/to/prompt.txt （若留空则不传）
SYSTEM_PROMPT_TEXT=""
# TEMPLATE_YAML can point to a chat template yaml (with {{ system_prompt }} placeholder)
TEMPLATE_YAML=""             # e.g., /path/to/template.yaml

# Python executables
PYTHON_BIN="python3"

# Paths to scripts
BEAM_SCRIPT="/home/nfs/u2023-zlb/CauDefend/src/ccd/inference/beam_structured_infer.py"
CONVERT_SCRIPT="/home/nfs/u2023-zlb/CauDefend/scripts/convert_clean_struct_to_eval.py"
EVAL_CPP="/home/nfs/u2023-zlb/CauDefend/scripts/eval_cpp_local.py"
EVAL_JAVA="/home/nfs/u2023-zlb/CauDefend/scripts/eval_java_local.py"
EVAL_PY="/home/nfs/u2023-zlb/CauDefend/scripts/eval_python_local.py"

#########################
# Helpers               #
#########################
latest_epoch_dir() {
  local root="$1"
  # If root itself is an epoch dir, just return it.
  if [[ -d "$root" && "$root" == *epoch* ]]; then
    echo "$root"
    return
  fi
  # Otherwise pick the last by name sort.
  local cand
  cand=$(ls -1d "${root}"/epoch* 2>/dev/null | sort | tail -n1 || true)
  if [[ -n "$cand" ]]; then
    echo "$cand"
  fi
}

ensure_path() {
  local p="$1"
  if [[ ! -f "$p" && ! -d "$p" ]]; then
    echo "Missing path: $p" >&2
    exit 1
  fi
}

run_clean() {
  local lang="$1"
  local infile="$2"
  local base="$3"
  local lora="$4"
  local tag="$5"

  ensure_path "$infile"
  ensure_path "$base"
  ensure_path "$lora"
  ensure_path "$BEAM_SCRIPT"

  local out_dir="${OUT_ROOT}/${tag}"
  mkdir -p "$out_dir"

  local cleaned_struct="${out_dir}/${lang}_clean_struct.jsonl"
  local cleaned_eval="${out_dir}/${lang}_eval.jsonl"
    echo "[${tag}/${lang}] cleaning -> ${cleaned_struct}"
  $PYTHON_BIN "$BEAM_SCRIPT" \
    --input "$infile" \
    --output "$cleaned_struct" \
    --model "$base" \
    --base-model "$base" \
    --peft-adapter "$lora" \
    --dtype "$DTYPE" \
    --device-map "$DEVICE_MAP" \
      ${TEMPLATE_YAML:+--template-yaml "$TEMPLATE_YAML"} \
      ${SYSTEM_PROMPT_TEXT:+--system-prompt-text "$SYSTEM_PROMPT_TEXT"} \
      ${SYSTEM_PROMPT_TEXT_FILE:+--system-prompt-text "$(cat "$SYSTEM_PROMPT_TEXT_FILE")"} \
    $GEN_ARGS

  echo "[${tag}/${lang}] converting to cleaned_variants -> ${cleaned_eval}"
  $PYTHON_BIN "$CONVERT_SCRIPT" --input "$cleaned_struct" --output "$cleaned_eval"

  echo "[${tag}/${lang}] done: ${cleaned_eval}"
}

run_eval() {
  local lang="$1"
  local infile="$2"
  local log_file="$3"
  local script=""
  case "$lang" in
    cpp) script="$EVAL_CPP" ;;
    java) script="$EVAL_JAVA" ;;
    python) script="$EVAL_PY" ;;
    *) echo "Unknown lang: $lang" >&2; exit 1 ;;
  esac
  ensure_path "$script"
  echo "[${lang}] evaluating -> ${log_file}"
  $PYTHON_BIN "$script" --input_file "$infile" > "$log_file" 2>&1 || true
}

#########################
# Main                  #
#########################

main() {
  # pick latest epoch for each lora root
  local lora_q25
  lora_q25=$(latest_epoch_dir "$LORA_ROOT_Q25")
  local lora_q3
  lora_q3=$(latest_epoch_dir "$LORA_ROOT_Q3")

  if [[ -z "$lora_q25" || -z "$lora_q3" ]]; then
    echo "Could not find epoch directories under LORA roots. Please adjust." >&2
    exit 1
  fi

  echo "Selected LoRA dirs:"
  echo " - Qwen2.5: $lora_q25"
  echo " - Qwen3-4B: $lora_q3"

  # configs to run
  declare -a MODELS=(
    "qwen25_latest|$MODEL_BASE_Q25|$lora_q25"
    "qwen3_latest|$MODEL_BASE_Q3|$lora_q3"
  )

  declare -a TASKS=(
    "cpp|$POISON_CPP"
    "java|$POISON_JAVA"
    "python|$POISON_PY"
  )

  for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r tag base lora <<<"$model_entry"
    for task_entry in "${TASKS[@]}"; do
      IFS='|' read -r lang infile <<<"$task_entry"
      source /home/nfs/u2023-zlb/miniconda3/bin/activate lmfty
      run_clean "$lang" "$infile" "$base" "$lora" "$tag"
      cleaned_eval="${OUT_ROOT}/${tag}/${lang}_eval.jsonl"
      log_file="${OUT_ROOT}/${tag}/${lang}_eval.log"
      source /home/nfs/u2023-zlb/miniconda3/bin/activate eval
      run_eval "$lang" "$cleaned_eval" "$log_file"
      echo "[${tag}/${lang}] eval log: ${log_file}"
    done
  done

  echo "All done. Outputs under ${OUT_ROOT}."
}

main "$@"


