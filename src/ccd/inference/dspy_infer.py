#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch

from ccd.inference.engine import load_model_and_tokenizer, ModelConfig as _HFModelCfg, PromptConfig as _HfPromptCfg, GenerationConfig as _HfGenCfg, generate_for_text as _hf_generate


def _require_dspy():
    try:
        import dspy  # type: ignore
        return dspy
    except Exception as e:
        raise ImportError(
            "dspy is required but not installed. Install with: pip install dspy-ai"
        ) from e


@dataclass
class DSPyModelConfig:
    model: str
    dtype: str = "float16"  # "float16" | "bfloat16" | "auto"
    device_map: str = "auto"


@dataclass
class DSPyGenConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = False
    n: int = 1
    num_beams: int = 1
    early_stopping: bool = True


def _torch_dtype_from_str(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return "auto"


_LM_CACHE: Dict[str, Any] = {}
_HF_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _lm_cache_key(cfg: DSPyModelConfig) -> str:
    return json.dumps({"model": cfg.model, "dtype": cfg.dtype, "device_map": cfg.device_map}, sort_keys=True)

def _hf_cache_key(cfg: DSPyModelConfig) -> str:
    return _lm_cache_key(cfg)

def get_or_load_hf(cfg: DSPyModelConfig) -> Tuple[Any, Any]:
    k = _hf_cache_key(cfg)
    if k in _HF_CACHE:
        return _HF_CACHE[k]
    model, tok = load_model_and_tokenizer(
        _HFModelCfg(model=cfg.model, dtype=cfg.dtype, device_map=cfg.device_map, trust_remote_code=False)
    )
    _HF_CACHE[k] = (model, tok)
    return model, tok

def get_or_load_lm(cfg: DSPyModelConfig):
    dspy = _require_dspy()
    k = _lm_cache_key(cfg)
    if k in _LM_CACHE:
        return _LM_CACHE[k]
    torch_dtype = _torch_dtype_from_str(cfg.dtype)
    # Preferred: native DSPy HF local wrappers (if available in installed version)
    try:
        if hasattr(dspy, "HFModel"):
            lm = dspy.HFModel(
                model=cfg.model,
                model_kwargs={"dtype": torch_dtype, "device_map": cfg.device_map},
            )
            dspy.settings.configure(lm=lm)
            _LM_CACHE[k] = lm
            return lm
        for cname in ("HFLM", "HFLocal", "HF"):
            if hasattr(dspy, cname):
                Cls = getattr(dspy, cname)
                try:
                    lm = Cls(model=cfg.model, model_kwargs={"dtype": torch_dtype, "device_map": cfg.device_map})
                except TypeError:
                    # Some variants may accept kwargs directly
                    lm = Cls(model=cfg.model, dtype=torch_dtype, device_map=cfg.device_map)
                dspy.settings.configure(lm=lm)
                _LM_CACHE[k] = lm
                return lm
    except Exception:
        pass

    # Fallback: implement a minimal local HF LM for DSPy using transformers
    model, tok = load_model_and_tokenizer(
        _HFModelCfg(model=cfg.model, dtype=cfg.dtype, device_map=cfg.device_map, trust_remote_code=False)
    )

    class _LocalHFLM(dspy.LM):  # type: ignore
        def __init__(self, model_obj, tok_obj):
            super().__init__(model="local-transformers")
            self._model = model_obj
            self._tok = tok_obj
            try:
                self._model_name = getattr(model_obj, "name_or_path", None) or "local-transformers"
            except Exception:
                self._model_name = "local-transformers"

        def _render_messages(self, prompt: Optional[str], messages: Optional[List[Dict[str, str]]]) -> str:
            if isinstance(prompt, str) and prompt.strip():
                return prompt
            if isinstance(messages, list) and messages:
                parts: List[str] = []
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    parts.append(f"{role}:\n{content}")
                return "\n\n".join(parts)
            return ""

        def _generate_many(self, text_prompt: str, **kwargs) -> List[str]:
            max_new_tokens = int(kwargs.get("max_new_tokens") or kwargs.get("max_tokens") or 512)
            temperature = float(kwargs.get("temperature", 0.7))
            top_p = float(kwargs.get("top_p", 0.95))
            do_sample = bool(kwargs.get("do_sample", False))
            num_beams = int(kwargs.get("num_beams", 1))
            n = int(kwargs.get("n", 1))
            num_return_sequences = max(1, n)
            inputs = self._tok(text_prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self._tok.pad_token_id,
                    eos_token_id=self._tok.eos_token_id,
                )
            seqs = out if isinstance(out, (list, tuple)) else out
            # HF returns tensor; decode all sequences
            if hasattr(seqs, "sequences"):
                seq_data = seqs.sequences
            else:
                seq_data = seqs
            texts: List[str] = []
            try:
                for i in range(len(seq_data)):
                    texts.append(self._tok.decode(seq_data[i], skip_special_tokens=True))
            except Exception:
                # Fallback: single sequence
                try:
                    texts = [self._tok.decode(seq_data[0], skip_special_tokens=True)]
                except Exception:
                    texts = [text_prompt]
            return texts

        # dspy BaseLM 调用链将走 __call__ -> forward，我们覆盖 forward 避免 litellm 路径
        def forward(self, prompt: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, **kwargs):
            text_prompt = self._render_messages(prompt, messages)
            texts = self._generate_many(text_prompt, **kwargs)
            # Return an object with `.choices` attribute, as expected by dspy
            class _Msg:
                def __init__(self, t: str):
                    self.content = t
            class _Choice:
                def __init__(self, t: str, idx: int):
                    self.message = _Msg(t)
                    self.text = t
                    self.index = idx
                    self.finish_reason = "stop"
            class _Resp:
                def __init__(self, ts: List[str], model_name: str):
                    self.choices = [_Choice(t, i) for i, t in enumerate(ts)]
                    # Provide minimal usage to satisfy dspy response parser
                    self.usage = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                    self.model = model_name
                    self.id = "chatcmpl-local"
                    self.object = "chat.completion"
                    self.created = int(time.time())
            return _Resp(texts, self._model_name)

        # 兼容可能被调用的 basic_request
        def basic_request(self, prompt: str, **kwargs) -> str:
            return self._generate_once(prompt, **kwargs)

    lm = _LocalHFLM(model, tok)
    dspy.settings.configure(lm=lm)
    _LM_CACHE[k] = lm
    return lm


def unload_lm(cfg: DSPyModelConfig) -> bool:
    k = _lm_cache_key(cfg)
    lm = _LM_CACHE.pop(k, None)
    if lm is None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False
    try:
        del lm
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return True


def _build_signature(signature_mode: str):
    dspy = _require_dspy()

    class BaseDefense(dspy.Signature):
        unsafe_code = dspy.InputField(desc="Original code with potential backdoors")
        analysis = dspy.OutputField(desc="Reasoning process identifying the backdoor")
        secure_code = dspy.OutputField(desc="Final sanitized code variants")

    class CodeCompletion(dspy.Signature):
        prompt = dspy.InputField(desc="Problem statement / function prompt")
        code = dspy.OutputField(desc="Completed implementation for the prompt")

    class CustomCompletion(dspy.Signature):
        prompt = dspy.InputField(desc="Fully rendered custom prompt")
        completion = dspy.OutputField(desc="Model output for custom prompt")

    # Freeform will bypass DSPy and use raw HF generate
    mode = (signature_mode or "completion").lower()
    if mode in ("defense", "base_defense", "security"):
        return BaseDefense, "defense"
    if mode in ("custom", "prompt"):
        return CustomCompletion, "custom"
    if mode in ("freeform", "raw", "text"):
        return None, "freeform"
    return CodeCompletion, "completion"


def _call_predictor(
    predictor: Any,
    signature_mode: str,
    input_text: str,
    gen: DSPyGenConfig,
) -> Tuple[List[str], Optional[List[str]]]:
    dspy = _require_dspy()
    cfg_keys = {
        "max_new_tokens": gen.max_new_tokens,
        "max_tokens": gen.max_new_tokens,
        "temperature": gen.temperature,
        "top_p": gen.top_p,
        "do_sample": gen.do_sample,
        "n": max(1, int(gen.n)),
        "num_beams": max(1, int(gen.num_beams)),
        "early_stopping": bool(gen.early_stopping),
    }
    config = {k: v for k, v in cfg_keys.items() if v is not None}

    responses: List[str] = []
    analyses: List[str] = []

    def _extract_single(res_obj) -> Tuple[str, Optional[str]]:
        if signature_mode == "defense":
            code_val = getattr(res_obj, "secure_code", None)
            ana_val = getattr(res_obj, "analysis", None)
            return (str(code_val) if code_val is not None else ""), (str(ana_val) if ana_val is not None else None)
        if signature_mode == "custom":
            comp_val = getattr(res_obj, "completion", None)
            return (str(comp_val) if comp_val is not None else ""), None
        code_val = getattr(res_obj, "code", None)
        return (str(code_val) if code_val is not None else ""), None

    try:
        if signature_mode == "defense":
            res = predictor(unsafe_code=input_text, config=config)
        elif signature_mode == "custom":
            res = predictor(prompt=input_text, config=config)
        else:
            res = predictor(prompt=input_text, config=config)
        if hasattr(res, "completions") and isinstance(res.completions, list) and res.completions:
            for c in res.completions:
                code_str, ana = _extract_single(c)
                if code_str is not None:
                    responses.append(code_str)
                    if ana is not None:
                        analyses.append(ana)
        else:
            code_str, ana = _extract_single(res)
            if code_str is not None:
                responses.append(code_str)
                if ana is not None:
                    analyses.append(ana)
        if not responses and config.get("n", 1) > 1:
            tries = max(1, int(config.get("n", 1)))
            for _ in range(tries):
                if signature_mode == "defense":
                    r = predictor(unsafe_code=input_text)
                elif signature_mode == "custom":
                    r = predictor(prompt=input_text)
                else:
                    r = predictor(prompt=input_text)
                code_str, ana = _extract_single(r)
                if code_str:
                    responses.append(code_str)
                    if ana is not None:
                        analyses.append(ana)
    except TypeError:
        if signature_mode == "defense":
            res = predictor(unsafe_code=input_text)
        elif signature_mode == "custom":
            res = predictor(prompt=input_text)
        else:
            res = predictor(prompt=input_text)
        code_str, ana = _extract_single(res)
        if code_str:
            responses.append(code_str)
            if ana is not None:
                analyses.append(ana)
    except Exception:
        raise

    return responses, (analyses if analyses else None)


def predict_single(
    model_cfg: DSPyModelConfig,
    signature_mode: str,
    input_text: str,
    gen_cfg: DSPyGenConfig,
    custom_prompt_text: Optional[str] = None,
    custom_vars: Optional[Dict[str, Any]] = None,
    extract_code: bool = False,
) -> Tuple[List[str], Optional[List[str]], int]:
    Sig, mode = _build_signature(signature_mode)
    responses: List[str] = []
    analyses: Optional[List[str]] = None
    t0 = time.time()
    effective_input = input_text
    if mode == "custom" and isinstance(custom_prompt_text, str) and custom_prompt_text.strip():
        mapping = {}
        if isinstance(custom_vars, dict):
            mapping.update(custom_vars)
        # Always provide poisoned_code default
        mapping.setdefault("poisoned_code", input_text)
        try:
            effective_input = custom_prompt_text.format(**mapping)
        except Exception:
            effective_input = custom_prompt_text
    if mode == "freeform":
        model, tok = get_or_load_hf(model_cfg)
        hf_gen = _HfGenCfg(
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=gen_cfg.do_sample,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            num_beams=gen_cfg.num_beams,
            num_return_sequences=gen_cfg.n if gen_cfg.n and gen_cfg.n > 0 else 1,
            num_beam_groups=1,
            diversity_penalty=0.0,
            seed=123456,
        )
        pr = _HfPromptCfg(template_yaml=None, system_prompt=None if False else None)  # plain prompt
        texts, weights, decoded = _hf_generate(model, tok, effective_input, pr, hf_gen)
        responses = texts
    else:
        lm = get_or_load_lm(model_cfg)
        dspy = _require_dspy()
        predictor = dspy.ChainOfThought(Sig)
        cands, analyses = _call_predictor(predictor, mode, effective_input, gen_cfg)
        responses = cands
    if extract_code:
        try:
            import re
            extracted: List[str] = []
            pat = re.compile(r"```[a-zA-Z0-9_+-]*\\n([\\s\\S]*?)```", re.MULTILINE)
            for t in responses:
                m = pat.search(t)
                extracted.append(m.group(1).strip() if m else t)
            responses = extracted
        except Exception:
            pass
    ms = int((time.time() - t0) * 1000)
    return responses, analyses, ms


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


def predict_dataset(
    input_path: str,
    output_path: str,
    field: str,
    model_cfg: DSPyModelConfig,
    signature_mode: str,
    gen_cfg: DSPyGenConfig,
    emit_flat: bool = True,
    write_mode: str = "generation",
    output_field: Optional[str] = None,
    limit: int = 0,
    custom_prompt_text: Optional[str] = None,
    custom_vars: Optional[Dict[str, Any]] = None,
    extract_code: bool = False,
    combine_fields: bool = False,
    prompt_field: Optional[str] = None,
    output_prompt_field: Optional[str] = None,
    output_code_field: Optional[str] = None,
) -> Dict[str, Any]:
    rows = read_jsonl(input_path, limit=limit)
    Sig, mode = _build_signature(signature_mode)
    if mode != "freeform":
        lm = get_or_load_lm(model_cfg)
        dspy = _require_dspy()
        predictor = dspy.ChainOfThought(Sig)
    else:
        model, tok = get_or_load_hf(model_cfg)
        hf_gen = _HfGenCfg(
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=gen_cfg.do_sample,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            num_beams=gen_cfg.num_beams,
            num_return_sequences=gen_cfg.n if gen_cfg.n and gen_cfg.n > 0 else 1,
            num_beam_groups=1,
            diversity_penalty=0.0,
            seed=123456,
        )
        pr = _HfPromptCfg(template_yaml=None, system_prompt=None)

    import re, json as _json

    def _extract_between(text: str, start_marker: str, end_marker: str) -> Optional[str]:
        si = text.find(start_marker)
        if si < 0:
            return None
        si += len(start_marker)
        ei = text.find(end_marker, si)
        if ei < 0:
            return None
        seg = text[si:ei]
        if isinstance(seg, str):
            if seg.startswith("\n"):
                seg = seg[1:]
            if seg.endswith("\n"):
                seg = seg[:-1]
        return seg

    def _split_output(out_text: str, fallback_prompt: str) -> Tuple[str, str]:
        try:
            obj = _json.loads(out_text)
            if isinstance(obj, dict):
                if "prompt" in obj and "code" in obj:
                    return str(obj.get("prompt") or ""), str(obj.get("code") or "")
                if "secure_code" in obj:
                    return fallback_prompt, str(obj.get("secure_code") or "")
                if "completion" in obj:
                    return fallback_prompt, str(obj.get("completion") or "")
        except Exception:
            pass
        for pstart, pend, cstart, cend in [
            ("# CCD_PROMPT_START", "# CCD_PROMPT_END", "# CCD_CODE_START", "# CCD_CODE_END"),
            ("PROMPT_START", "PROMPT_END", "CODE_START", "CODE_END"),
        ]:
            p = _extract_between(out_text, pstart, pend)
            c = _extract_between(out_text, cstart, cend)
            if p is not None and c is not None:
                return p, c
        m = re.search(r"```[a-zA-Z0-9_+-]*\n([\s\S]*?)```", out_text, re.MULTILINE)
        if m:
            return (fallback_prompt, m.group(1).strip())
        return (fallback_prompt, out_text)

    out_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for obj in rows:
        text = obj.get(field, None)
        if not isinstance(text, str) or not text.strip():
            continue
        effective_input = text
        prompt_val = ""
        if mode in ("custom", "freeform") and isinstance(custom_prompt_text, str) and custom_prompt_text.strip():
            mapping = {}
            if isinstance(custom_vars, dict):
                mapping.update(custom_vars)
            if combine_fields and isinstance(prompt_field, str) and prompt_field.strip():
                prompt_val = str(obj.get(prompt_field, "") or "")
                mapping.setdefault("prompt", prompt_val)
                mapping.setdefault("code", text)
            mapping.setdefault("poisoned_code", text)
            try:
                effective_input = custom_prompt_text.format(**mapping)
            except Exception:
                effective_input = custom_prompt_text
        if mode == "freeform":
            texts, weights, decoded = _hf_generate(model, tok, effective_input, pr, hf_gen)
            cands = texts
            analyses = None
        else:
            cands, analyses = _call_predictor(predictor, mode, effective_input, gen_cfg)
        if extract_code:
            pat = re.compile(r"```[a-zA-Z0-9_+-]*\n([\s\S]*?)```", re.MULTILINE)
            tmp = []
            for t in cands:
                m = pat.search(t)
                tmp.append(m.group(1).strip() if m else t)
            cands = tmp
        effective_flat = emit_flat or (gen_cfg.n and gen_cfg.n > 1)
        if write_mode == "generation":
            dst = output_field.strip() if (isinstance(output_field, str) and output_field.strip()) else "generation"
            if effective_flat:
                for i, cand in enumerate(cands):
                    out_obj = dict(obj)
                    if combine_fields and (output_prompt_field or output_code_field):
                        p_out, c_out = _split_output(cand, prompt_val)
                        if output_prompt_field and output_prompt_field.strip():
                            out_obj[output_prompt_field] = p_out
                        if output_code_field and output_code_field.strip():
                            out_obj[output_code_field] = c_out
                        out_obj[dst] = c_out if extract_code else cand
                    else:
                        out_obj[dst] = cand
                    out_obj["completion_id"] = i
                    if analyses is not None and i < len(analyses or []):
                        out_obj["analysis"] = analyses[i]
                    out_rows.append(out_obj)
            else:
                out_obj = dict(obj)
                resp = cands[0] if cands else text
                if combine_fields and (output_prompt_field or output_code_field):
                    p_out, c_out = _split_output(resp, prompt_val)
                    if output_prompt_field and output_prompt_field.strip():
                        out_obj[output_prompt_field] = p_out
                    if output_code_field and output_code_field.strip():
                        out_obj[output_code_field] = c_out
                    out_obj[dst] = c_out if extract_code else resp
                else:
                    out_obj[dst] = resp
                if analyses:
                    out_obj["analysis"] = analyses[0]
                out_rows.append(out_obj)
        else:
            if effective_flat:
                for i, cand in enumerate(cands):
                    out_obj = dict(obj)
                    if combine_fields and (output_prompt_field or output_code_field):
                        p_out, c_out = _split_output(cand, prompt_val)
                        if output_prompt_field and output_prompt_field.strip():
                            out_obj[output_prompt_field] = p_out
                        out_obj[field] = c_out if extract_code else cand
                        if output_code_field and output_code_field.strip():
                            out_obj[output_code_field] = c_out
                    else:
                        out_obj[field] = cand
                    out_obj["completion_id"] = i
                    if analyses is not None and i < len(analyses or []):
                        out_obj["analysis"] = analyses[i]
                    out_rows.append(out_obj)
            else:
                out_obj = dict(obj)
                resp = cands[0] if cands else text
                if combine_fields and (output_prompt_field or output_code_field):
                    p_out, c_out = _split_output(resp, prompt_val)
                    if output_prompt_field and output_prompt_field.strip():
                        out_obj[output_prompt_field] = p_out
                    out_obj[field] = c_out if extract_code else resp
                    if output_code_field and output_code_field.strip():
                        out_obj[output_code_field] = c_out
                else:
                    out_obj[field] = resp
                if analyses:
                    out_obj["analysis"] = analyses[0]
                out_rows.append(out_obj)

    write_jsonl(output_path, out_rows)
    ms = int((time.time() - t0) * 1000)
    return {
        "total": len(rows),
        "output_path": output_path,
        "elapsed_ms": ms,
        "preview": out_rows[: min(5, len(out_rows))],
    }


