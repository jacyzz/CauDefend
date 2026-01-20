#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


@dataclass
class ModelConfig:
    model: str
    dtype: str = "float16"  # "float16" | "bfloat16" | "auto"
    device_map: str = "auto"
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = False
    use_safetensors: bool = False
    base_model: str = ""        # optional for PEFT
    peft_adapter: str = ""      # optional for PEFT
    peft_merge: bool = False


@dataclass
class PromptConfig:
    template_yaml: Optional[str] = None
    system_prompt_text: Optional[str] = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    num_beams: int = 1
    num_return_sequences: int = 1
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    seed: int = 123456


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


def _dir_is_peft_adapter(path: str) -> bool:
    try:
        files = set(os.listdir(path))
    except Exception:
        return False
    return any(n in files for n in ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"])


def _torch_dtype_from_str(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return "auto"


def _load_with_peft(base: str, adapter: str, model_kwargs: Dict[str, Any], trust_remote_code: bool, merge: bool):
    try:
        from peft import PeftModel  # type: ignore
    except Exception as e:
        raise ImportError("peft is required to load LoRA adapters. Install: pip install peft") from e
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=trust_remote_code, padding_side="left")
    mdl = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)
    mdl = PeftModel.from_pretrained(mdl, adapter)
    if merge:
        mdl = mdl.merge_and_unload()
    return mdl, tok


def load_model_and_tokenizer(cfg: ModelConfig):
    torch_dtype = _torch_dtype_from_str(cfg.dtype)
    model_kwargs: Dict[str, Any] = dict(
        device_map=cfg.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.trust_remote_code,
        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        use_safetensors=cfg.use_safetensors,
        # Enable KV cache to speed up generation, especially for long outputs
        use_cache=True,
    )

    # Explicit base+adapter has highest priority
    if cfg.base_model and cfg.peft_adapter:
        model, tokenizer = _load_with_peft(
            cfg.base_model, cfg.peft_adapter, model_kwargs, cfg.trust_remote_code, cfg.peft_merge
        )
    else:
        want = cfg.model
        if _dir_has_weights(want):
            try:
                files = set(os.listdir(want))
            except Exception:
                files = set()
            if ("model.safetensors.index.json" in files) or any(fn.endswith(".safetensors") for fn in files):
                model_kwargs["use_safetensors"] = True
            model = AutoModelForCausalLM.from_pretrained(want, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(want, trust_remote_code=cfg.trust_remote_code, padding_side="left")
        elif _dir_is_peft_adapter(want):
            base_from_cfg = ""
            try:
                cfg_path = os.path.join(want, "adapter_config.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        adapter_cfg = json.load(f)
                        base_from_cfg = adapter_cfg.get("base_model_name_or_path") or adapter_cfg.get("base_model_name")
            except Exception:
                base_from_cfg = ""
            base_path = cfg.base_model or base_from_cfg
            if not base_path:
                raise OSError(
                    f"Detected PEFT adapter at {want}, but base model is unknown. "
                    f"Provide ModelConfig.base_model or ensure adapter_config.json contains base_model_name_or_path."
                )
            model, tokenizer = _load_with_peft(base_path, want, model_kwargs, cfg.trust_remote_code, cfg.peft_merge)
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
    return model, tokenizer


def load_system_prompt(cfg: PromptConfig) -> str:
    base = (cfg.system_prompt_text or "").strip()
    if cfg.template_yaml:
        if yaml is None:
            raise ImportError("PyYAML is required to load template yaml. Install: pip install pyyaml")
        data = yaml.safe_load(open(cfg.template_yaml, "r", encoding="utf-8"))
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
    # Prefer FIRST response block; models sometimes repeat the header multiple times
    first = full_text.find(key)
    if first < 0:
        content = full_text.strip()
    else:
        tail = full_text[first + len(key):]
        # If there is a second "### Response:", cut before it
        second = tail.find(key)
        if second >= 0:
            tail = tail[:second]
        content = tail.strip()

    # Strip code fences if any
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


def softmax_scores(scores: torch.Tensor) -> torch.Tensor:
    try:
        return torch.nn.functional.softmax(scores, dim=0)
    except Exception:
        return scores


def generate_for_text(
    model,
    tokenizer,
    input_text: str,
    prompt_cfg: PromptConfig,
    gen_cfg: GenerationConfig,
) -> Tuple[List[str], Optional[List[float]], List[str]]:
    torch.manual_seed(gen_cfg.seed)
    system_prompt = load_system_prompt(prompt_cfg)
    prompt = build_prompt(input_text, system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=gen_cfg.max_new_tokens,
        do_sample=gen_cfg.do_sample,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        num_beams=gen_cfg.num_beams,
        num_return_sequences=gen_cfg.num_return_sequences,
        num_beam_groups=gen_cfg.num_beam_groups,
        diversity_penalty=gen_cfg.diversity_penalty,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Sanitize incompatible combos: if no grouping, disable diversity
    if (gen_kwargs.get("num_beam_groups") or 1) <= 1 and (gen_kwargs.get("diversity_penalty") or 0.0) > 0.0:
        gen_kwargs["diversity_penalty"] = 0.0
    with torch.no_grad():
        try:
            outputs = model.generate(**gen_kwargs)
        except RuntimeError as e:
            msg = str(e)
            # Known HF bug: bfloat16 + group beam search may raise dtype mismatch
            if "Index put requires the source and destination dtypes match" in msg:
                # Fallback A: disable group beam search
                tried = False
                if gen_cfg.num_beam_groups and gen_cfg.num_beam_groups > 1:
                    gen_kwargs["num_beam_groups"] = 1
                    gen_kwargs["diversity_penalty"] = 0.0
                    try:
                        outputs = model.generate(**gen_kwargs)
                        tried = True
                    except Exception:
                        pass
                if not tried:
                    # Fallback B: cast model to float16 and retry
                    try:
                        first_param = next(model.parameters(), None)
                        if first_param is not None and first_param.dtype == torch.bfloat16:
                            model.to(torch.float16)
                        outputs = model.generate(**gen_kwargs)
                    except Exception:
                        raise
            else:
                raise
    seqs = outputs.sequences
    seq_scores = outputs.sequences_scores if hasattr(outputs, "sequences_scores") else None
    weights = None
    if seq_scores is not None:
        weights = softmax_scores(seq_scores).tolist()
    decoded = [tokenizer.decode(s, skip_special_tokens=True) for s in seqs]

    responses: List[str] = []
    for t in decoded:
        resp = extract_response(t)
        if not resp.strip():
            # Fallback to tail or original text
            pos = t.rfind("### Response:")
            tail = t[pos + len("### Response:"):].strip() if pos >= 0 else t.strip()
            resp = tail if tail else input_text
        responses.append(resp)
    return responses, weights, decoded


