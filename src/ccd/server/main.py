from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import gc
import threading
import uuid

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

from ccd.ist.transfer import StyleTransfer
from ccd.ist.styles import STYLE_DICT
from ccd.inference.engine import (
    ModelConfig as HFModelConfig,
    PromptConfig as HfPromptConfig,
    GenerationConfig as HfGenConfig,
    load_model_and_tokenizer,
    generate_for_text,
    load_system_prompt as eng_load_system_prompt,
    build_prompt as eng_build_prompt,
    softmax_scores as eng_softmax,
)
from ccd.inference.dspy_infer import (
    DSPyModelConfig as _DSPyModelConfig,
    DSPyGenConfig as _DSPyGenConfig,
    predict_single as _dspy_predict_single,
    predict_dataset as _dspy_predict_dataset,
    unload_lm as _dspy_unload,
)

# optional tqdm for progress in dataset inference
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

app = FastAPI(title="CCD Backend", version="0.1.0")

# Allow local dev UIs by default; tighten in production if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransformTextReq(BaseModel):
    language: str = Field(..., description="e.g. 'python','java','c','cpp','javascript','go','php'")
    code: str
    strategy: str = Field("fixed", description="'fixed' | 'random'")
    styles: Optional[List[str]] = None
    poison_min: int = 2
    poison_max: int = 3
    avoid_similar: bool = True
    seed: Optional[int] = None


class TransformTextResp(BaseModel):
    converted_code: str
    applied_styles: List[str]
    syntax_ok: bool
    processing_time_ms: int
    log: List[str] = []


class TransformDatasetReq(BaseModel):
    input_path: str
    output_path: str
    language: str
    code_field: str
    # Humaneval-style: optionally combine prompt+code for transform, then split back
    combine_fields: bool = False
    prompt_field: Optional[str] = None
    output_prompt_field: Optional[str] = None
    output_code_field: Optional[str] = None
    id_field: Optional[str] = None
    backup_field: Optional[str] = None
    strategy: str = "fixed"  # fixed | random
    styles: Optional[List[str]] = None
    # 可选：用于随机策略的候选风格池
    poison_candidates: Optional[List[str]] = None
    poison_min: int = 2
    poison_max: int = 3
    avoid_similar: bool = True
    limit: int = 0
    seed: Optional[int] = None
    syntax_check: bool = False


class TransformDatasetResp(BaseModel):
    total: int
    changed: int
    success: int
    output_path: str
    preview: List[Dict[str, Any]] = []
    log: List[Dict[str, Any]] = []

# --------- IST async progress tracking ---------
_IST_TASKS: Dict[str, Dict[str, Any]] = {}
_IST_TASKS_LOCK = threading.Lock()

class TransformDatasetAsyncResp(BaseModel):
    task_id: str
    total: int
    status: str = "started"

class ISTProgressResp(BaseModel):
    task_id: str
    status: str
    current: int
    total: int
    percent: float
    error: Optional[str] = None
    result: Optional[TransformDatasetResp] = None

# -------- Inference (HF beam-search) --------

class InferModelCfg(BaseModel):
    model: str
    dtype: str = "float16"  # "float16" | "bfloat16" | "auto"
    device_map: str = "auto"
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = False
    use_safetensors: bool = False
    base_model: Optional[str] = None
    peft_adapter: Optional[str] = None
    peft_merge: bool = False


class InferPromptCfg(BaseModel):
    template_yaml: Optional[str] = None
    system_prompt_text: Optional[str] = None


class InferGenParams(BaseModel):
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    num_beams: int = 1
    num_return_sequences: int = 1
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    seed: int = 123456


class InferTextReq(BaseModel):
    input_text: str
    model: InferModelCfg
    prompt: InferPromptCfg = InferPromptCfg()
    gen: InferGenParams = InferGenParams()
    return_decoded: bool = False
    unload_after: bool = False


class InferTextResp(BaseModel):
    candidates: List[str]
    scores: Optional[List[float]] = None
    elapsed_ms: int
    log: List[str] = []
    decoded: Optional[List[str]] = None


class InferDatasetReq(BaseModel):
    input_path: str
    output_path: str
    field: str
    # When write_mode == "generation", write candidates into this field instead of hardcoded "generation"
    output_field: Optional[str] = None
    model: InferModelCfg
    prompt: InferPromptCfg = InferPromptCfg()
    gen: InferGenParams = InferGenParams()
    emit_flat: bool = True
    write_mode: str = Field("generation", description="'generation' | 'overwrite'")
    limit: int = 0
    # Progress options (printed on server console; frontend remains unchanged)
    progress: bool = True
    progress_every: int = 10
    unload_after: bool = False
    # Optional: merge prompt + code for input; then split model output back
    combine_fields: bool = False
    prompt_field: Optional[str] = None
    output_prompt_field: Optional[str] = None
    output_code_field: Optional[str] = None
    extract_code: bool = False


class InferDatasetResp(BaseModel):
    total: int
    output_path: str
    preview: List[Dict[str, Any]] = []
    elapsed_ms: int
    log: List[str] = []

# ----- New structured dataset API models -----

class InputBuilder(BaseModel):
    mode: str = Field("merge", description="'merge' | 'single'")
    # when mode=single
    field: Optional[str] = None
    # when mode=merge
    fields: Optional[List[str]] = None  # default to ['declaration','canonical_solution'] if None
    separator: str = "\n\n"
    prefix: str = ""
    suffix: str = ""
    id_field: str = "task_id"


class OutputSchema(BaseModel):
    mode: str = Field("structured_variants", description="'structured_variants' | 'single_field'")
    field: str = "output"  # for structured_variants, array field name; for single_field, target field
    emit_flat: bool = False
    preset: str = "humaneval_structured"  # current preset
    trace_analysis: str = ""  # constant value per variant
    extract_sections: bool = True  # try to parse [Trace Analysis] and [Sanitized Code]


class InferDatasetStructuredReq(BaseModel):
    input_path: str
    output_path: str
    input_builder: InputBuilder = InputBuilder()
    output_schema: OutputSchema = OutputSchema()
    model: InferModelCfg
    prompt: InferPromptCfg = InferPromptCfg()
    gen: InferGenParams = InferGenParams()
    limit: int = 0
    unload_after: bool = False


_MODEL_CACHE: Dict[str, Any] = {}

def _model_cache_key(cfg: InferModelCfg) -> str:
    key = {
        "model": cfg.model,
        "dtype": cfg.dtype,
        "device_map": cfg.device_map,
        "trust_remote_code": cfg.trust_remote_code,
        "low_cpu_mem_usage": cfg.low_cpu_mem_usage,
        "use_safetensors": cfg.use_safetensors,
        "base_model": cfg.base_model or "",
        "peft_adapter": cfg.peft_adapter or "",
        "peft_merge": cfg.peft_merge,
    }
    return json.dumps(key, sort_keys=True)

def _get_or_load_model(cfg: InferModelCfg):
    k = _model_cache_key(cfg)
    if k in _MODEL_CACHE:
        return _MODEL_CACHE[k]
    hf_cfg = HFModelConfig(
        model=cfg.model,
        dtype=cfg.dtype,
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        use_safetensors=cfg.use_safetensors,
        base_model=cfg.base_model or "",
        peft_adapter=cfg.peft_adapter or "",
        peft_merge=cfg.peft_merge,
    )
    model, tok = load_model_and_tokenizer(hf_cfg)
    _MODEL_CACHE[k] = (model, tok)
    return model, tok


def _release_model(cfg: InferModelCfg) -> bool:
    """Remove model from cache and try to free GPU memory."""
    k = _model_cache_key(cfg)
    item = _MODEL_CACHE.pop(k, None)
    if item is None:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass
        return False
    model, tok = item
    try:
        if hasattr(model, "cpu"):
            model.cpu()
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass
    try:
        del tok
    except Exception:
        pass
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    return True


def _read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                # skip malformed lines
                continue
            rows.append(obj)
            if limit and len(rows) >= limit:
                break
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _choose_random_styles(
    language: str,
    rng,
    min_n: int,
    max_n: int,
    avoid_similar: bool,
    pool: Optional[List[str]] = None,
) -> List[str]:
    """Pick a few styles randomly, optionally from a custom pool."""
    default_pool = ["-1.1", "-3.1", "0.5", "7.2", "8.1"]
    choices = list(pool) if (pool and len(pool) > 0) else list(default_pool)
    rng.shuffle(choices)
    target = max(min_n, min(max_n, max_n))  # ensure sane

    def group_key(s: str) -> str:
        return s.split(".")[0]

    selected: List[str] = []
    used_groups = set()
    for st in choices:
        if len(selected) >= target:
            break
        if avoid_similar and group_key(st) in used_groups:
            continue
        selected.append(st)
        used_groups.add(group_key(st))
    if not selected and choices:
        selected = [choices[0]]
    return selected


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

# ---------- Inference: JSONL inspection (fields + preview) ----------

@app.get("/api/infer/inspect_jsonl")
def infer_inspect_jsonl(path: str = Query(...), limit: int = Query(5, ge=1, le=100)) -> Dict[str, Any]:
    p = Path(path)
    rows = _read_jsonl(p, limit=limit)
    fields = set()
    for r in rows:
        try:
            fields.update(list(r.keys()))
        except Exception:
            pass
    return {
        "path": str(p),
        "count_preview": len(rows),
        "fields": sorted(list(fields)),
        "preview": rows,
    }


@app.get("/api/ist/styles")
def list_styles() -> Dict[str, List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    for code, (stype, ssub, pre) in STYLE_DICT.items():
        items.append(
            {
                "code": code,
                "family": code.split(".")[0] if "." in code else code,
                "type": stype,
                "subtype": ssub,
                "prepare": pre,
            }
        )
    items.sort(key=lambda x: (int(x["family"]) if x["family"].isdigit() else 999, x["code"]))
    return {"styles": items}


@app.post("/api/ist/transform_text", response_model=TransformTextResp)
def transform_text(req: TransformTextReq) -> TransformTextResp:
    import random

    if not req.code or not req.language:
        raise HTTPException(status_code=400, detail="language and code are required")
    rng = random.Random(req.seed)

    st = StyleTransfer(req.language)
    styles_to_apply: List[str]
    if req.strategy == "fixed":
        if not req.styles:
            raise HTTPException(status_code=400, detail="styles required for fixed strategy")
        styles_to_apply = [s.strip() for s in (req.styles or []) if s.strip()]
    else:
        styles_to_apply = _choose_random_styles(
            req.language,
            rng,
            req.poison_min,
            req.poison_max,
            req.avoid_similar,
        )

    start = time.time()
    current = req.code
    applied: List[str] = []
    for s in styles_to_apply:
        try:
            new_code, ok = st.transfer(styles=[s], code=current)
            if ok and isinstance(new_code, str) and new_code != current:
                current = new_code
                applied.append(s)
        except Exception as e:
            # skip broken style
            continue
    syntax_ok = bool(st.check_syntax(current))
    elapsed_ms = int((time.time() - start) * 1000)
    return TransformTextResp(
        converted_code=current,
        applied_styles=applied,
        syntax_ok=syntax_ok,
        processing_time_ms=elapsed_ms,
        log=[f"strategy={req.strategy}", f"styles={styles_to_apply}", f"applied={applied}"],
    )


@app.post("/api/ist/transform_dataset", response_model=TransformDatasetResp)
def transform_dataset(req: TransformDatasetReq) -> TransformDatasetResp:
    import random

    input_path = Path(req.input_path)
    output_path = Path(req.output_path)
    rows = _read_jsonl(input_path, limit=req.limit or 0)
    if not rows:
        raise HTTPException(status_code=400, detail="No rows to process")

    rng = random.Random(req.seed)
    st = StyleTransfer(req.language)

    def _comment_prefix(lang: str) -> str:
        m = {
            "python": "#",
            "py": "#",
            "javascript": "//",
            "js": "//",
            "java": "//",
            "c": "//",
            "cpp": "//",
            "c++": "//",
            "c_sharp": "//",
            "csharp": "//",
            "go": "//",
            "php": "//",
        }
        return m.get(lang.lower(), "#")

    total = 0
    changed = 0
    success = 0
    syntax_failed = 0
    out_rows: List[Dict[str, Any]] = []
    run_log: List[Dict[str, Any]] = []

    for idx, obj in enumerate(rows):
        total += 1
        code_val = obj.get(req.code_field, "")
        prompt_val = None
        if req.combine_fields:
            pf = req.prompt_field.strip() if isinstance(req.prompt_field, str) else "prompt"
            prompt_val = obj.get(pf, "")
            if not isinstance(prompt_val, str):
                prompt_val = ""
        if not isinstance(code_val, str):
            code_val = ""
        if (not code_val.strip()) and (not (req.combine_fields and isinstance(prompt_val, str) and prompt_val.strip())):
            run_log.append({"index": idx, "status": "skipped", "reason": "missing_or_invalid_code_and_prompt"})
            out_rows.append(obj)
            continue

        # decide styles
        if req.strategy == "fixed":
            if not req.styles:
                run_log.append({"index": idx, "status": "skipped", "reason": "no_styles"})
                out_rows.append(obj)
                continue
            styles_to_apply = [s.strip() for s in (req.styles or []) if s.strip()]
        else:
            styles_to_apply = _choose_random_styles(
                req.language,
                rng,
                req.poison_min,
                req.poison_max,
                req.avoid_similar,
                req.poison_candidates,
            )

        # Apply transform
        combined_mode = bool(req.combine_fields)
        current = code_val
        combined_before = None
        pfx = _comment_prefix(req.language)
        P_START = f"{pfx} CCD_PROMPT_START"
        P_END = f"{pfx} CCD_PROMPT_END"
        C_START = f"{pfx} CCD_CODE_START"
        C_END = f"{pfx} CCD_CODE_END"
        if combined_mode:
            prompt_text = prompt_val or ""
            combined_before = f"{P_START}\n{prompt_text}\n{P_END}\n{C_START}\n{code_val}\n{C_END}\n"
            current = combined_before
        applied: List[str] = []
        ok_any = False
        for s in styles_to_apply:
            try:
                # Ensure deterministic behavior for random-based transforms
                try:
                    import random as _rand  # local alias to avoid shadowing
                    base_seed = int(req.seed) if isinstance(req.seed, (int, float)) else 0
                    _rand.seed(f"{base_seed}-{idx}-{s}")
                except Exception:
                    pass
                new_code, ok = st.transfer(styles=[s], code=current)
                if ok and isinstance(new_code, str) and new_code != current:
                    current = new_code
                    applied.append(s)
                    ok_any = True
            except Exception:
                pass

        out_obj = dict(obj)
        if not combined_mode:
            if req.backup_field:
                out_obj[req.backup_field] = code_val
            out_obj[req.code_field] = current
        else:
            # Split back by markers; if markers lost, fall back to original fields
            out_prompt = prompt_val or ""
            out_code = code_val
            try:
                def _extract_between(text: str, start_marker: str, end_marker: str) -> Optional[str]:
                    si = text.find(start_marker)
                    if si < 0:
                        return None
                    si += len(start_marker)
                    ei = text.find(end_marker, si)
                    if ei < 0:
                        return None
                    # trim single leading newline if present
                    segment = text[si:ei]
                    if segment.startswith("\n"):
                        segment = segment[1:]
                    if segment.endswith("\n"):
                        segment = segment[:-1]
                    return segment

                extr_prompt = _extract_between(current, P_START, P_END)
                extr_code = _extract_between(current, C_START, C_END)
                if isinstance(extr_prompt, str):
                    out_prompt = extr_prompt
                if isinstance(extr_code, str):
                    out_code = extr_code
            except Exception:
                pass
            # write outputs
            dst_prompt_field = (
                req.output_prompt_field.strip()
                if isinstance(req.output_prompt_field, str) and req.output_prompt_field.strip()
                else (req.prompt_field if isinstance(req.prompt_field, str) and req.prompt_field.strip() else "prompt")
            )
            dst_code_field = (
                req.output_code_field.strip()
                if isinstance(req.output_code_field, str) and req.output_code_field.strip()
                else req.code_field
            )
            if req.backup_field:
                out_obj[req.backup_field] = code_val
            out_obj[dst_prompt_field] = out_prompt
            out_obj[dst_code_field] = out_code
        out_obj.setdefault("ist", {})
        syntax_ok = True
        if req.syntax_check:
            try:
                syntax_ok = bool(st.check_syntax(current))
            except Exception:
                syntax_ok = False
            if not syntax_ok:
                syntax_failed += 1
        out_obj["ist"].update(
            {
                "language": req.language,
                "attempted_styles": styles_to_apply,
                "applied_styles": applied,
                "success": ok_any,
                "syntax_checked": req.syntax_check,
                "syntax_ok": syntax_ok,
                "combined": combined_mode,
            }
        )
        if ok_any:
            success += 1
        if not combined_mode:
            if current != code_val:
                changed += 1
        else:
            if combined_before is not None and current != combined_before:
                changed += 1
        out_rows.append(out_obj)
        if idx < 20:
            run_log.append(
                {
                    "index": idx,
                    "status": "ok" if ok_any else "no-change",
                    "applied_styles": applied,
                    "changed": (current != code_val) if not combined_mode else (combined_before is not None and current != combined_before),
                    "syntax_checked": req.syntax_check,
                    "syntax_ok": syntax_ok,
                    "combined": combined_mode,
                }
            )

    _write_jsonl(output_path, out_rows)
    preview = out_rows[: min(5, len(out_rows))]
    return TransformDatasetResp(
        total=total, changed=changed, success=success, output_path=str(output_path), preview=preview, log=run_log
    )

def _ist_worker(task_id: str, req: TransformDatasetReq):
    import random
    try:
        input_path = Path(req.input_path)
        output_path = Path(req.output_path)
        rows = _read_jsonl(input_path, limit=req.limit or 0)
        if not rows:
            raise RuntimeError("No rows to process")
        rng = random.Random(req.seed)
        st = StyleTransfer(req.language)

        def _comment_prefix(lang: str) -> str:
            m = {
                "python": "#", "py": "#", "javascript": "//", "js": "//", "java": "//",
                "c": "//", "cpp": "//", "c++": "//", "c_sharp": "//", "csharp": "//",
                "go": "//", "php": "//",
            }
            return m.get(lang.lower(), "#")

        total = len(rows)
        changed = 0
        success = 0
        syntax_failed = 0
        out_rows: List[Dict[str, Any]] = []
        run_log: List[Dict[str, Any]] = []

        for idx, obj in enumerate(rows):
            with _IST_TASKS_LOCK:
                if task_id in _IST_TASKS:
                    _IST_TASKS[task_id]["current"] = idx
            code_val = obj.get(req.code_field, "")
            prompt_val = None
            if req.combine_fields:
                pf = req.prompt_field.strip() if isinstance(req.prompt_field, str) else "prompt"
                prompt_val = obj.get(pf, "")
                if not isinstance(prompt_val, str):
                    prompt_val = ""
            if not isinstance(code_val, str):
                code_val = ""
            if (not code_val.strip()) and (not (req.combine_fields and isinstance(prompt_val, str) and prompt_val.strip())):
                out_rows.append(obj)
                continue
            if req.strategy == "fixed":
                styles_to_apply = [s.strip() for s in (req.styles or []) if s.strip()] if req.styles else []
            else:
                styles_to_apply = _choose_random_styles(
                    req.language, rng, req.poison_min, req.poison_max, req.avoid_similar, req.poison_candidates
                )
            combined_mode = bool(req.combine_fields)
            current = code_val
            combined_before = None
            pfx = _comment_prefix(req.language)
            P_START = f"{pfx} CCD_PROMPT_START"
            P_END = f"{pfx} CCD_PROMPT_END"
            C_START = f"{pfx} CCD_CODE_START"
            C_END = f"{pfx} CCD_CODE_END"
            if combined_mode:
                prompt_text = prompt_val or ""
                combined_before = f"{P_START}\n{prompt_text}\n{P_END}\n{C_START}\n{code_val}\n{C_END}\n"
                current = combined_before
            applied: List[str] = []
            ok_any = False
            for s in styles_to_apply:
                try:
                    # Deterministic seeding for random-based transforms
                    try:
                        import random as _rand
                        base_seed = int(req.seed) if isinstance(req.seed, (int, float)) else 0
                        _rand.seed(f"{base_seed}-{idx}-{s}")
                    except Exception:
                        pass
                    new_code, ok = st.transfer(styles=[s], code=current)
                    if ok and isinstance(new_code, str) and new_code != current:
                        current = new_code
                        applied.append(s)
                        ok_any = True
                except Exception:
                    pass
            out_obj = dict(obj)
            if not combined_mode:
                if req.backup_field:
                    out_obj[req.backup_field] = code_val
                out_obj[req.code_field] = current
            else:
                out_prompt = prompt_val or ""
                out_code = code_val
                try:
                    def _extract_between(text: str, start_marker: str, end_marker: str) -> Optional[str]:
                        si = text.find(start_marker)
                        if si < 0:
                            return None
                        si += len(start_marker)
                        ei = text.find(end_marker, si)
                        if ei < 0:
                            return None
                        segment = text[si:ei]
                        if segment.startswith("\n"):
                            segment = segment[1:]
                        if segment.endswith("\n"):
                            segment = segment[:-1]
                        return segment
                    extr_prompt = _extract_between(current, P_START, P_END)
                    extr_code = _extract_between(current, C_START, C_END)
                    if isinstance(extr_prompt, str):
                        out_prompt = extr_prompt
                    if isinstance(extr_code, str):
                        out_code = extr_code
                except Exception:
                    pass
                dst_prompt_field = (
                    req.output_prompt_field.strip()
                    if isinstance(req.output_prompt_field, str) and req.output_prompt_field.strip()
                    else (req.prompt_field if isinstance(req.prompt_field, str) and req.prompt_field.strip() else "prompt")
                )
                dst_code_field = (
                    req.output_code_field.strip()
                    if isinstance(req.output_code_field, str) and req.output_code_field.strip()
                    else req.code_field
                )
                if req.backup_field:
                    out_obj[req.backup_field] = code_val
                out_obj[dst_prompt_field] = out_prompt
                out_obj[dst_code_field] = out_code
            out_obj.setdefault("ist", {})
            syntax_ok = True
            if req.syntax_check:
                try:
                    syntax_ok = bool(st.check_syntax(current))
                except Exception:
                    syntax_ok = False
                if not syntax_ok:
                    syntax_failed += 1
            out_obj["ist"].update(
                {
                    "language": req.language,
                    "attempted_styles": styles_to_apply,
                    "applied_styles": applied,
                    "success": ok_any,
                    "syntax_checked": req.syntax_check,
                    "syntax_ok": syntax_ok,
                    "combined": combined_mode,
                }
            )
            if ok_any:
                success += 1
            if not combined_mode:
                if current != code_val:
                    changed += 1
            else:
                if combined_before is not None and current != combined_before:
                    changed += 1
            out_rows.append(out_obj)

        _write_jsonl(output_path, out_rows)
        preview = out_rows[: min(5, len(out_rows))]
        result = TransformDatasetResp(
            total=total, changed=changed, success=success, output_path=str(output_path), preview=preview, log=[]
        )
        with _IST_TASKS_LOCK:
            _IST_TASKS[task_id].update(
                {"status": "done", "current": total, "result": result}
            )
    except BaseException as e:
        with _IST_TASKS_LOCK:
            if task_id in _IST_TASKS:
                _IST_TASKS[task_id].update({"status": "error", "error": str(e)})

@app.post("/api/ist/transform_dataset_async", response_model=TransformDatasetAsyncResp)
def transform_dataset_async(req: TransformDatasetReq) -> TransformDatasetAsyncResp:
    rows = _read_jsonl(Path(req.input_path), limit=req.limit or 0)
    if not rows:
        raise HTTPException(status_code=400, detail="No rows to process")
    task_id = uuid.uuid4().hex
    with _IST_TASKS_LOCK:
        _IST_TASKS[task_id] = {"status": "running", "current": 0, "total": len(rows)}
    t = threading.Thread(target=_ist_worker, args=(task_id, req), daemon=True)
    t.start()
    return TransformDatasetAsyncResp(task_id=task_id, total=len(rows))

@app.get("/api/ist/progress", response_model=ISTProgressResp)
def ist_progress(task_id: str = Query(...)) -> ISTProgressResp:
    with _IST_TASKS_LOCK:
        info = _IST_TASKS.get(task_id)
    if not info:
        raise HTTPException(status_code=404, detail="task_id not found")
    status = info.get("status", "running")
    current = int(info.get("current", 0))
    total = int(info.get("total", 1))
    percent = (current / total * 100.0) if total else 0.0
    error = info.get("error")
    result = info.get("result")
    return ISTProgressResp(
        task_id=task_id, status=status, current=current, total=total, percent=percent, error=error, result=result
    )


@app.get("/api/ist/dataset_schema")
def dataset_schema(path: str = Query(...), preview: int = Query(5)) -> Dict[str, Any]:
    p = Path(path)
    rows = _read_jsonl(p, limit=preview)
    fields = set()
    for r in rows:
        fields.update(r.keys())
    return {"path": str(p), "count_preview": len(rows), "fields": sorted(list(fields)), "preview": rows}


@app.get("/api/ist/record")
def get_record(
    path: str = Query(...),
    index: int = Query(..., ge=0),
    code_field: str = Query(...),
    backup_field: Optional[str] = Query(None),
) -> Dict[str, Any]:
    p = Path(path)
    rows = _read_jsonl(p, limit=0)
    if index < 0 or index >= len(rows):
        raise HTTPException(status_code=404, detail="Index out of range")
    row = rows[index]
    before = row.get(backup_field) if backup_field else None
    after = row.get(code_field)
    return {"index": index, "record": row, "before": before, "after": after}


@app.post("/api/infer/generate", response_model=InferTextResp)
def infer_generate(req: InferTextReq) -> InferTextResp:
    if not req.input_text or not req.model or not req.model.model:
        raise HTTPException(status_code=400, detail="input_text and model.model are required")
    model, tok = _get_or_load_model(req.model)
    gen_cfg = HfGenConfig(
        max_new_tokens=req.gen.max_new_tokens,
        do_sample=req.gen.do_sample,
        temperature=req.gen.temperature,
        top_p=req.gen.top_p,
        num_beams=req.gen.num_beams,
        num_return_sequences=req.gen.num_return_sequences,
        num_beam_groups=req.gen.num_beam_groups,
        diversity_penalty=req.gen.diversity_penalty,
        seed=req.gen.seed,
    )
    pr_cfg = HfPromptConfig(
        template_yaml=req.prompt.template_yaml,
        system_prompt_text=req.prompt.system_prompt_text,
    )
    t0 = time.time()
    responses, weights, decoded = generate_for_text(model, tok, req.input_text, pr_cfg, gen_cfg)
    ms = int((time.time() - t0) * 1000)
    # Sanitize weights for JSON (avoid NaN/Inf causing no-response serialization errors)
    safe_weights: Optional[List[float]] = None
    if weights is not None:
        try:
            safe_weights = [
                (float(w) if (isinstance(w, (int, float)) and math.isfinite(float(w))) else 0.0)
                for w in weights
            ]
        except Exception:
            safe_weights = None
    if req.unload_after:
        try:
            _release_model(req.model)
        except Exception:
            pass
    return InferTextResp(
        candidates=responses,
        scores=safe_weights,
        elapsed_ms=ms,
        log=[
            f"num_beams={req.gen.num_beams}",
            f"num_return_sequences={req.gen.num_return_sequences}",
            f"num_beam_groups={req.gen.num_beam_groups}",
            f"diversity_penalty={req.gen.diversity_penalty}",
            f"do_sample={req.gen.do_sample}",
            f"temperature={req.gen.temperature}",
            f"top_p={req.gen.top_p}",
            f"max_new_tokens={req.gen.max_new_tokens}",
            f"unload_after={req.unload_after}",
        ],
        decoded=decoded if req.return_decoded else None,
    )


@app.post("/api/infer/dataset", response_model=InferDatasetResp)
def infer_dataset(req: InferDatasetReq) -> InferDatasetResp:
    # Reuse engine's robust text generator per record to avoid dtype/group-beam issues.
    from ccd.inference.beam_infer import read_jsonl as _read, write_jsonl as _write

    model, tok = _get_or_load_model(req.model)
    gen_cfg = HfGenConfig(
        max_new_tokens=req.gen.max_new_tokens,
        do_sample=req.gen.do_sample,
        temperature=req.gen.temperature,
        top_p=req.gen.top_p,
        num_beams=req.gen.num_beams,
        num_return_sequences=req.gen.num_return_sequences,
        num_beam_groups=req.gen.num_beam_groups,
        diversity_penalty=req.gen.diversity_penalty,
        seed=req.gen.seed,
    )
    pr_cfg = HfPromptConfig(
        template_yaml=req.prompt.template_yaml,
        system_prompt_text=req.prompt.system_prompt_text,
    )
    rows = _read(req.input_path, limit=req.limit or 0)
    out_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    iterator = tqdm(rows, total=len(rows), desc="InferDataset", unit="sample") if req.progress else rows
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
        if seg.startswith("\n"):
            seg = seg[1:]
        if seg.endswith("\n"):
            seg = seg[:-1]
        return seg

    def _split_output(out_text: str, fallback_prompt: str) -> Tuple[str, str]:
        # 1) JSON with keys
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
        # 2) Markers like CCD_PROMPT_START/END and CCD_CODE_START/END
        for pstart, pend, cstart, cend in [
            ("# CCD_PROMPT_START", "# CCD_PROMPT_END", "# CCD_CODE_START", "# CCD_CODE_END"),
            ("PROMPT_START", "PROMPT_END", "CODE_START", "CODE_END"),
        ]:
            p = _extract_between(out_text, pstart, pend)
            c = _extract_between(out_text, cstart, cend)
            if p is not None and c is not None:
                return p, c
        # 3) First fenced code block
        m = re.search(r"```[a-zA-Z0-9_+-]*\n([\s\S]*?)```", out_text, re.MULTILINE)
        if m:
            return (fallback_prompt, m.group(1).strip())
        # 4) Fallback: treat entire text as code
        return (fallback_prompt, out_text)

    for idx, obj in enumerate(iterator):
        code = obj.get(req.field, None)
        if not isinstance(code, str) or not code.strip():
            continue
        input_text = code
        prompt_val = ""
        if req.combine_fields and isinstance(req.prompt_field, str) and req.prompt_field.strip():
            prompt_val = str(obj.get(req.prompt_field, "") or "")
            # Simple concatenation; you may also enforce a template on frontend
            input_text = f"{prompt_val}\n\n{code}"
        responses, weights, _decoded = generate_for_text(model, tok, input_text, pr_cfg, gen_cfg)
        if req.progress and req.progress_every > 0 and (idx + 1) % req.progress_every == 0:
            try:
                # update tqdm postfix if available
                if hasattr(iterator, "set_postfix_str"):
                    iterator.set_postfix_str(f"processed={idx+1}")
            except Exception:
                pass

        effective_flat = req.emit_flat or (req.gen.num_return_sequences and req.gen.num_return_sequences > 1)
        if req.write_mode == "generation":
            dst_field = (
                req.output_field.strip()
                if (isinstance(req.output_field, str) and req.output_field.strip())
                else "generation"
            )
            if effective_flat:
                for i, resp in enumerate(responses):
                    out_obj = dict(obj)
                    if req.combine_fields and (req.output_prompt_field or req.output_code_field):
                        p_out, c_out = _split_output(resp, prompt_val)
                        if req.output_prompt_field and req.output_prompt_field.strip():
                            out_obj[req.output_prompt_field] = p_out
                        if req.output_code_field and req.output_code_field.strip():
                            out_obj[req.output_code_field] = c_out
                        out_obj[dst_field] = c_out if req.extract_code else resp
                    else:
                        out_obj[dst_field] = resp
                    out_obj["completion_id"] = i
                    if weights is not None and i < len(weights):
                        out_obj["variant_score"] = weights[i]
                    out_rows.append(out_obj)
            else:
                out_obj = dict(obj)
                resp = responses[0] if responses else code
                if req.combine_fields and (req.output_prompt_field or req.output_code_field):
                    p_out, c_out = _split_output(resp, prompt_val)
                    if req.output_prompt_field and req.output_prompt_field.strip():
                        out_obj[req.output_prompt_field] = p_out
                    if req.output_code_field and req.output_code_field.strip():
                        out_obj[req.output_code_field] = c_out
                    out_obj[dst_field] = c_out if req.extract_code else resp
                else:
                    out_obj[dst_field] = resp
                out_rows.append(out_obj)
        else:
            if effective_flat:
                for i, resp in enumerate(responses):
                    out_obj = dict(obj)
                    if req.combine_fields and (req.output_prompt_field or req.output_code_field):
                        p_out, c_out = _split_output(resp, prompt_val)
                        if req.output_prompt_field and req.output_prompt_field.strip():
                            out_obj[req.output_prompt_field] = p_out
                        # overwrite code field with parsed code
                        out_obj[req.field] = c_out if req.extract_code else resp
                        if req.output_code_field and req.output_code_field.strip():
                            out_obj[req.output_code_field] = c_out
                    else:
                        out_obj[req.field] = resp
                    out_obj["completion_id"] = i
                    if weights is not None and i < len(weights):
                        out_obj["variant_score"] = weights[i]
                    out_rows.append(out_obj)
            else:
                out_obj = dict(obj)
                resp = responses[0] if responses else code
                if req.combine_fields and (req.output_prompt_field or req.output_code_field):
                    p_out, c_out = _split_output(resp, prompt_val)
                    if req.output_prompt_field and req.output_prompt_field.strip():
                        out_obj[req.output_prompt_field] = p_out
                    out_obj[req.field] = c_out if req.extract_code else resp
                    if req.output_code_field and req.output_code_field.strip():
                        out_obj[req.output_code_field] = c_out
                else:
                    out_obj[req.field] = resp
                out_rows.append(out_obj)

    _write(req.output_path, out_rows)
    ms = int((time.time() - t0) * 1000)
    preview = out_rows[: min(5, len(out_rows))]
    if req.unload_after:
        try:
            _release_model(req.model)
        except Exception:
            pass
    return InferDatasetResp(
        total=len(rows),
        output_path=req.output_path,
        preview=preview,
        elapsed_ms=ms,
        log=[
            f"num_beams={req.gen.num_beams}",
            f"num_return_sequences={req.gen.num_return_sequences}",
            f"emit_flat={req.emit_flat}",
            f"write_mode={req.write_mode}",
            f"output_field={req.output_field or ('generation' if req.write_mode == 'generation' else req.field)}",
            f"processed={len(rows)}",
            f"unload_after={req.unload_after}",
        ],
    )

@app.post("/api/infer/dataset_structured", response_model=InferDatasetResp)
def infer_dataset_structured(req: InferDatasetStructuredReq) -> InferDatasetResp:
    model, tok = _get_or_load_model(req.model)
    gen_cfg = HfGenConfig(
        max_new_tokens=req.gen.max_new_tokens,
        do_sample=req.gen.do_sample,
        temperature=req.gen.temperature,
        top_p=req.gen.top_p,
        num_beams=req.gen.num_beams,
        num_return_sequences=req.gen.num_return_sequences,
        num_beam_groups=req.gen.num_beam_groups,
        diversity_penalty=req.gen.diversity_penalty,
        seed=req.gen.seed,
    )
    pr_cfg = HfPromptConfig(
        template_yaml=req.prompt.template_yaml,
        system_prompt_text=req.prompt.system_prompt_text,
    )
    rows = _read_jsonl(Path(req.input_path), limit=req.limit or 0)
    out_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    # defaults for merge
    default_fields = ["declaration", "canonical_solution"]

    import re

    def _first_fenced_code_block(s: str) -> Optional[str]:
        m = re.search(r"```[a-zA-Z0-9_+-]*\n([\s\S]*?)```", s, re.MULTILINE)
        if m:
            return m.group(1).strip()
        return None

    def _extract_analysis_code(txt: str) -> Tuple[str, str]:
        """
        Try to split model output into analysis and code.
        Supported markers:
          - [Trace Analysis] ... [Sanitized Code] ... (preferred)
          - 'Trace Analysis:' heading, optional 'Sanitized Code:' heading
          - fenced code block: take preceding text as analysis
        Fallback: analysis='', code=txt
        """
        t = txt.strip()
        # Normalize windows line endings
        t = t.replace("\r\n", "\n")
        # 1) [Trace Analysis] ... [Sanitized Code]
        m1 = re.search(r"(?is)\[trace\s*analysis\](.*)\[sanitized\s*code\](.*)$", t)
        if m1:
            analysis = m1.group(1).strip()
            tail = m1.group(2).strip()
            code = _first_fenced_code_block(tail) or tail
            return analysis, code
        # 2) 'Trace Analysis:' and then fenced code or 'Sanitized Code:'
        m2 = re.search(r"(?is)trace\s*analysis\s*:\s*(.*)$", t)
        if m2:
            after = m2.group(1)
            # Prefer explicit 'Sanitized Code:' split
            m2b = re.search(r"(?is)(.*)sanitized\s*code\s*:\s*(.*)$", after)
            if m2b:
                analysis = m2b.group(1).strip()
                tail = m2b.group(2).strip()
                code = _first_fenced_code_block(tail) or tail
                return analysis, code
            # Else use first fenced code block
            code_block = _first_fenced_code_block(after)
            if code_block is not None:
                # analysis = text before block
                head = after.split("```", 1)[0].strip()
                return head, code_block
        # 3) Only fenced code block
        only = _first_fenced_code_block(t)
        if only is not None:
            head = t.split("```", 1)[0].strip()
            return head, only
        # 4) Fallback
        return "", t

    for obj in rows:
        # Build input
        ib = req.input_builder
        if ib.mode == "single":
            fld = (ib.field or "").strip()
            src = str(obj.get(fld, "") or "")
            if not src.strip():
                continue
            input_text = src
        else:
            fields = ib.fields if (isinstance(ib.fields, list) and len(ib.fields) > 0) else default_fields
            parts: List[str] = []
            for f in fields:
                val = obj.get(f, None)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
            if not parts:
                continue
            input_text = ib.separator.join(parts)
        if ib.prefix:
            input_text = f"{ib.prefix}{input_text}"
        if ib.suffix:
            input_text = f"{input_text}{ib.suffix}"

        # Generate
        responses, weights, _decoded = generate_for_text(model, tok, input_text, pr_cfg, gen_cfg)

        # Compose output
        osch = req.output_schema
        if osch.mode == "single_field":
            out_obj = dict(obj)
            out_obj[osch.field] = responses if osch.emit_flat else (responses[0] if responses else "")
            out_rows.append(out_obj)
            continue

        # structured_variants preset: humaneval_structured
        variants: List[Dict[str, Any]] = []
        for i, resp in enumerate(responses, start=1):
            if osch.extract_sections:
                analysis, code_part = _extract_analysis_code(resp)
                trace_text = analysis if analysis.strip() else osch.trace_analysis
                variants.append(
                    {"variant": i, "trace_analysis": trace_text, "sanitized_code": code_part}
                )
            else:
                variants.append(
                    {"variant": i, "trace_analysis": osch.trace_analysis, "sanitized_code": resp}
                )
        out_obj: Dict[str, Any] = {
            "task_id": obj.get(ib.id_field, None),
            "canonical_solution": obj.get("canonical_solution", None),
            osch.field: variants,
        }
        # Include declaration only if present
        decl_val = obj.get("declaration", None)
        if isinstance(decl_val, str) and decl_val.strip():
            out_obj["declaration"] = decl_val
        out_rows.append(out_obj)

    _write_jsonl(Path(req.output_path), out_rows)
    ms = int((time.time() - t0) * 1000)
    preview = out_rows[: min(5, len(out_rows))]
    if req.unload_after:
        try:
            _release_model(req.model)
        except Exception:
            pass
    return InferDatasetResp(
        total=len(out_rows),
        output_path=req.output_path,
        preview=preview,
        elapsed_ms=ms,
        log=[
            f"preset={req.output_schema.preset}",
            f"structured_field={req.output_schema.field}",
            f"id_field={req.input_builder.id_field}",
            f"num_beams={req.gen.num_beams}",
            f"num_return_sequences={req.gen.num_return_sequences}",
            f"unload_after={req.unload_after}",
        ],
    )


class InferUnloadReq(BaseModel):
    model: InferModelCfg


@app.post("/api/infer/unload")
def infer_unload(req: InferUnloadReq) -> Dict[str, Any]:
    ok = _release_model(req.model)
    return {"ok": ok}


@app.post("/api/infer/unload_all")
def infer_unload_all() -> Dict[str, Any]:
    keys = list(_MODEL_CACHE.keys())
    any_ok = False
    for k in keys:
        try:
            model, tok = _MODEL_CACHE.pop(k)
            try:
                if hasattr(model, "cpu"):
                    model.cpu()
            except Exception:
                pass
            del model
            del tok
            any_ok = True
        except Exception:
            pass
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    return {"ok": any_ok, "cleared": True, "remaining": len(_MODEL_CACHE)}


class DSpyModelCfg(BaseModel):
    model: str
    dtype: str = "float16"
    device_map: str = "auto"


class DSpyGenParams(BaseModel):
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = False
    n: int = 1
    num_beams: int = 1
    early_stopping: bool = True


class DSpyTextReq(BaseModel):
    input_text: str
    signature_mode: str = "completion"  # "completion" | "defense" | "custom" | "freeform"
    model: DSpyModelCfg
    gen: DSpyGenParams = DSpyGenParams()
    unload_after: bool = False
    # Custom prompt support
    custom_prompt_text: Optional[str] = None
    custom_vars: Optional[Dict[str, Any]] = None
    extract_code: bool = False


class DSpyTextResp(BaseModel):
    candidates: List[str]
    analyses: Optional[List[str]] = None
    elapsed_ms: int


class DSpyDatasetReq(BaseModel):
    input_path: str
    output_path: str
    field: str
    signature_mode: str = "completion"  # "completion" | "defense" | "custom" | "freeform"
    model: DSpyModelCfg
    gen: DSpyGenParams = DSpyGenParams()
    emit_flat: bool = True
    write_mode: str = Field("generation", description="'generation' | 'overwrite'")
    output_field: Optional[str] = None
    limit: int = 0
    unload_after: bool = False
    # Custom prompt support
    custom_prompt_text: Optional[str] = None
    custom_vars: Optional[Dict[str, Any]] = None
    extract_code: bool = False
    # Combine prompt + code then split back if needed (for datasets like HumanEval)
    combine_fields: bool = False
    prompt_field: Optional[str] = None
    output_prompt_field: Optional[str] = None
    output_code_field: Optional[str] = None


class DSpyDatasetResp(BaseModel):
    total: int
    output_path: str
    elapsed_ms: int
    preview: List[Dict[str, Any]] = []


@app.post("/api/dspy/generate", response_model=DSpyTextResp)
def dspy_generate(req: DSpyTextReq) -> DSpyTextResp:
    if not req.input_text or not req.model or not req.model.model:
        raise HTTPException(status_code=400, detail="input_text and model.model are required")
    mdl = _DSPyModelConfig(model=req.model.model, dtype=req.model.dtype, device_map=req.model.device_map)
    gen = _DSPyGenConfig(
        max_new_tokens=req.gen.max_new_tokens,
        temperature=req.gen.temperature,
        top_p=req.gen.top_p,
        do_sample=req.gen.do_sample,
        n=req.gen.n,
        num_beams=req.gen.num_beams,
        early_stopping=req.gen.early_stopping,
    )
    cands, analyses, ms = _dspy_predict_single(
        mdl,
        req.signature_mode,
        req.input_text,
        gen,
        custom_prompt_text=req.custom_prompt_text,
        custom_vars=req.custom_vars,
        extract_code=req.extract_code,
    )
    if req.unload_after:
        try:
            _dspy_unload(mdl)
        except Exception:
            pass
    return DSpyTextResp(candidates=cands, analyses=analyses, elapsed_ms=ms)


@app.post("/api/dspy/dataset", response_model=DSpyDatasetResp)
def dspy_dataset(req: DSpyDatasetReq) -> DSpyDatasetResp:
    if not req.input_path or not req.output_path or not req.field:
        raise HTTPException(status_code=400, detail="input_path, output_path and field are required")
    mdl = _DSPyModelConfig(model=req.model.model, dtype=req.model.dtype, device_map=req.model.device_map)
    gen = _DSPyGenConfig(
        max_new_tokens=req.gen.max_new_tokens,
        temperature=req.gen.temperature,
        top_p=req.gen.top_p,
        do_sample=req.gen.do_sample,
        n=req.gen.n,
        num_beams=req.gen.num_beams,
        early_stopping=req.gen.early_stopping,
    )
    result = _dspy_predict_dataset(
        input_path=req.input_path,
        output_path=req.output_path,
        field=req.field,
        model_cfg=mdl,
        signature_mode=req.signature_mode,
        gen_cfg=gen,
        emit_flat=req.emit_flat,
        write_mode=req.write_mode,
        output_field=req.output_field,
        limit=req.limit or 0,
        custom_prompt_text=req.custom_prompt_text,
        custom_vars=req.custom_vars,
        extract_code=req.extract_code,
        combine_fields=req.combine_fields,
        prompt_field=req.prompt_field,
        output_prompt_field=req.output_prompt_field,
        output_code_field=req.output_code_field,
    )
    if req.unload_after:
        try:
            _dspy_unload(mdl)
        except Exception:
            pass
    return DSpyDatasetResp(
        total=result.get("total", 0),
        output_path=result.get("output_path", req.output_path),
        elapsed_ms=result.get("elapsed_ms", 0),
        preview=result.get("preview", []),
    )


