#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):
        return x

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def read_jsonl(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append(obj)
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_system_prompt(template_yaml: Optional[str], system_prompt_text: Optional[str]) -> str:
    base = system_prompt_text or ""
    if template_yaml:
        if yaml is None:
            raise ImportError("PyYAML is required to load template yaml. Install: pip install pyyaml")
        data = yaml.safe_load(open(template_yaml, "r", encoding="utf-8"))
        content = data.get("messages", [{}])[0].get("content", "")
        content = content.replace("{{ system_prompt }}", base).replace("{{system_prompt}}", base)
        return content
    return base


def build_prompt(input_text: str, system_prompt: str) -> str:
    header = (system_prompt or "").strip()
    if header:
        return f"{header}\n\n### Input:\n{input_text}\n\n### Response:"
    return f"### Input:\n{input_text}\n\n### Response:"


def extract_response(full_text: str) -> str:
    key = "### Response:"
    left = full_text.rfind(key)
    content = full_text[left + len(key):].strip() if left >= 0 else full_text.strip()

    if content.startswith("```"):
        lines = content.splitlines()
        rest = lines[1:]
        end_idx = None
        for i, line in enumerate(rest):
            if line.strip().startswith("```"):
                end_idx = i
                break
        if end_idx is not None:
            rest = rest[:end_idx]
        content = "\n".join(rest).strip()
    for marker in ("</s>", "[/INST]"):
        if content.endswith(marker):
            content = content[: -len(marker)].strip()
    return content


def extract_analysis_code(txt: str) -> (str, str):
    """
    Split model output into analysis and code.
    Rules (best-effort):
      1) [Trace Analysis] ... [Sanitized Code] ...
      2) Trace Analysis: ... Sanitized Code: ...
      3) First fenced code block as code; prefix as analysis
      4) Fallback: analysis="", code=txt
    """
    import re
    t = (txt or "").replace("\r\n", "\n")

    def first_fenced(s: str) -> str | None:
        m = re.search(r"```[a-zA-Z0-9_+\-]*\n([\s\S]*?)```", s, re.MULTILINE)
        if m:
            return m.group(1).strip()
        return None

    # 1) [Trace Analysis] ... [Sanitized Code] ...
    m1 = re.search(r"\[trace\s*analysis\](.*)\[sanitized\s*code\](.*)$", t, flags=re.IGNORECASE | re.DOTALL)
    if m1:
        analysis = m1.group(1).strip()
        tail = m1.group(2).strip()
        code = first_fenced(tail) or tail
        return analysis, code

    # 2) Trace Analysis: ... Sanitized Code:
    m2 = re.search(r"trace\s*analysis\s*:\s*(.*)$", t, flags=re.IGNORECASE | re.DOTALL)
    if m2:
        after = m2.group(1)
        m2b = re.search(r"(.*)sanitized\s*code\s*:\s*(.*)$", after, flags=re.IGNORECASE | re.DOTALL)
        if m2b:
            analysis = m2b.group(1).strip()
            tail = m2b.group(2).strip()
            code = first_fenced(tail) or tail
            return analysis, code
        block = first_fenced(after)
        if block:
            head = after.split("```", 1)[0].strip()
            return head, block

    # 3) Fenced code block only
    only = first_fenced(t)
    if only:
        head = t.split("```", 1)[0].strip()
        return head, only

    # 4) Fallback
    return "", t.strip()


def softmax_scores(scores: torch.Tensor) -> torch.Tensor:
    try:
        return torch.nn.functional.softmax(scores, dim=0)
    except Exception:
        return scores


def merge_declaration_and_code(declaration: Optional[str], canonical_solution: Optional[str]) -> Optional[str]:
    parts: List[str] = []
    if isinstance(declaration, str) and declaration.strip():
        parts.append(declaration.strip())
    if isinstance(canonical_solution, str) and canonical_solution.strip():
        parts.append(canonical_solution.strip())
    if not parts:
        return None
    return "\n\n".join(parts)


def main():
    p = argparse.ArgumentParser(description="HF beam-search inference producing structured output with merged inputs")
    # IO
    p.add_argument("--input", required=True, help="输入 JSONL 路径")
    p.add_argument("--output", required=True, help="输出 JSONL 路径")
    p.add_argument("--limit", type=int, default=0, help="处理条数，0 表示全部")
    # Field controls
    p.add_argument("--decl-field", default="declaration", help="声明字段名（默认：declaration）")
    p.add_argument("--code-field", default="canonical_solution", help="代码字段名（默认：canonical_solution）")
    p.add_argument("--output-field", default="output", help="输出候选列表字段名（默认：output）")
    # Model
    p.add_argument("--model", required=True, help="HF 模型名或本地路径")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "auto"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--low-cpu-mem-usage", action="store_true")
    p.add_argument("--use-safetensors", action="store_true")
    # Optional PEFT/LoRA support
    p.add_argument("--base-model", default="", help="可选：指定底座模型（用于 LoRA 适配器合并/加载）")
    p.add_argument("--peft-adapter", default="", help="可选：指定 LoRA/PEFT 适配器目录（与 --base-model 搭配）")
    p.add_argument("--peft-merge", action="store_true", help="加载后合并 LoRA 权重（占用更多显存，但推理更快）")
    # Prompt
    p.add_argument("--template-yaml", default="", help="可选：提示模板 YAML，使用 {{ system_prompt }} 占位符")
    p.add_argument("--system-prompt-text", default="", help="可选：系统提示文本，注入到模板占位符")
    # Generation
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--num-return-sequences", type=int, default=1)
    p.add_argument("--num-beam-groups", type=int, default=1)
    p.add_argument("--diversity-penalty", type=float, default=0.0)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=123456)

    args = p.parse_args()
    torch.manual_seed(args.seed)

    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto"

    model_kwargs = dict(
        device_map=args.device_map,
        dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        use_safetensors=args.use_safetensors,
        use_cache=False,
    )

    def _dir_has_weights(path: str) -> bool:
        names = {
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        }
        try:
            files = set(os.listdir(path))
        except Exception:
            return False
        if "model.safetensors.index.json" in files or "pytorch_model.bin.index.json" in files:
            return True
        return any(n in files for n in names)

    def _maybe_enable_safetensors(path: str) -> None:
        try:
            files = set(os.listdir(path))
        except Exception:
            return
        if "model.safetensors.index.json" in files or any(fn.endswith(".safetensors") for fn in files):
            model_kwargs["use_safetensors"] = True

    def _dir_is_peft_adapter(path: str) -> bool:
        try:
            files = set(os.listdir(path))
        except Exception:
            return False
        return any(n in files for n in ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"])

    def _load_with_peft(base: str, adapter: str):
        try:
            from peft import PeftModel  # type: ignore
        except Exception as e:
            raise ImportError("peft is required to load LoRA adapters. Install: pip install peft") from e
        _maybe_enable_safetensors(base)
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=args.trust_remote_code)
        mdl = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)
        mdl = PeftModel.from_pretrained(mdl, adapter)
        if args.peft_merge:
            mdl = mdl.merge_and_unload()
        return mdl, tok

    # Resolve model/tokenizer source
    model: Any
    tokenizer: Any
    if args.base_model and args.peft_adapter:
        model, tokenizer = _load_with_peft(args.base_model, args.peft_adapter)
    else:
        want = args.model
        if _dir_has_weights(want):
            try:
                files = set(os.listdir(want))
            except Exception:
                files = set()
            _maybe_enable_safetensors(want)
            model = AutoModelForCausalLM.from_pretrained(want, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(want, trust_remote_code=args.trust_remote_code)
        elif _dir_is_peft_adapter(want):
            base_from_cfg = ""
            try:
                cfg_path = os.path.join(want, "adapter_config.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        base_from_cfg = cfg.get("base_model_name_or_path") or cfg.get("base_model_name")
            except Exception:
                base_from_cfg = ""
            base_path = args.base_model or base_from_cfg
            if not base_path:
                raise OSError(
                    f"Detected PEFT adapter at {want}, but no base model provided. "
                    f"Please pass --base-model /path/to/base or ensure adapter_config.json contains base_model_name_or_path."
                )
            model, tokenizer = _load_with_peft(base_path, want)
        else:
            try:
                listing = ", ".join(os.listdir(want))
            except Exception:
                listing = "N/A"
            raise OSError(
                f"Model directory '{want}' does not contain recognized weights. "
                f"Expect model.safetensors / pytorch_model.bin or their index files. "
                f"Directory listing: {listing}"
            )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_prompt = load_system_prompt(args.template_yaml or None, args.system_prompt_text or None)

    rows = read_jsonl(args.input, limit=args.limit)
    out_rows: List[Dict[str, Any]] = []
    total_time = 0.0
    n = 0

    for obj in tqdm(rows, total=len(rows), desc="Generating", unit="sample"):
        merged_input = merge_declaration_and_code(
            obj.get(args.decl_field, None),
            obj.get(args.code_field, None),
        )
        if not isinstance(merged_input, str) or not merged_input.strip():
            continue

        prompt = build_prompt(merged_input, system_prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        start_t = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                num_beam_groups=args.num_beam_groups,
                diversity_penalty=args.diversity_penalty,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        dt = time.time() - start_t
        total_time += dt
        n += 1

        seqs = outputs.sequences
        decoded = [tokenizer.decode(s, skip_special_tokens=True) for s in seqs]

        responses: List[str] = []
        analyses: List[str] = []
        codes: List[str] = []
        for t in decoded:
            resp = extract_response(t)
            if not resp.strip():
                pos = t.rfind("### Response:")
                tail = t[pos + len("### Response:"):].strip() if pos >= 0 else t.strip()
                resp = tail if tail else (obj.get(args.code_field, "") or "")
            analysis, code = extract_analysis_code(resp)
            responses.append(resp)
            analyses.append(analysis)
            codes.append(code)

        out_obj = dict(obj)
        variants: List[Dict[str, Any]] = []
        for i, resp in enumerate(responses, start=1):
            analysis = analyses[i-1] if i-1 < len(analyses) else ""
            code = codes[i-1] if i-1 < len(codes) else resp
            variants.append(
                {
                    "variant": i,
                    "trace_analysis": analysis,
                    "sanitized_code": code,
                }
            )
        out_obj[args.output_field] = variants
        out_rows.append(out_obj)

    write_jsonl(args.output, out_rows)
    summary_path = str(Path(args.output).with_suffix("")) + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": n,
                "avg_time": (total_time / n) if n else 0.0,
                "num_beams": args.num_beams,
                "num_return_sequences": args.num_return_sequences,
                "num_beam_groups": args.num_beam_groups,
                "diversity_penalty": args.diversity_penalty,
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "model": args.model,
                "input": args.input,
                "output": args.output,
                "decl_field": args.decl_field,
                "code_field": args.code_field,
                "output_field": args.output_field,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote {len(out_rows)} records to {args.output}")


if __name__ == "__main__":
    main()


