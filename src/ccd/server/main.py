from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class TransformDatasetResp(BaseModel):
    total: int
    changed: int
    success: int
    output_path: str
    preview: List[Dict[str, Any]] = []
    log: List[Dict[str, Any]] = []

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
    model: InferModelCfg
    prompt: InferPromptCfg = InferPromptCfg()
    gen: InferGenParams = InferGenParams()
    emit_flat: bool = True
    write_mode: str = Field("generation", description="'generation' | 'overwrite'")
    limit: int = 0


class InferDatasetResp(BaseModel):
    total: int
    output_path: str
    preview: List[Dict[str, Any]] = []
    elapsed_ms: int
    log: List[str] = []


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

    total = 0
    changed = 0
    success = 0
    out_rows: List[Dict[str, Any]] = []
    run_log: List[Dict[str, Any]] = []

    for idx, obj in enumerate(rows):
        total += 1
        code_val = obj.get(req.code_field, "")
        if not isinstance(code_val, str) or not code_val.strip():
            run_log.append({"index": idx, "status": "skipped", "reason": "missing_or_invalid_code"})
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

        current = code_val
        applied: List[str] = []
        ok_any = False
        for s in styles_to_apply:
            try:
                new_code, ok = st.transfer(styles=[s], code=current)
                if ok and isinstance(new_code, str) and new_code != current:
                    current = new_code
                    applied.append(s)
                    ok_any = True
            except Exception:
                pass

        out_obj = dict(obj)
        if req.backup_field:
            out_obj[req.backup_field] = code_val
        out_obj[req.code_field] = current
        out_obj.setdefault("ist", {})
        out_obj["ist"].update(
            {
                "language": req.language,
                "attempted_styles": styles_to_apply,
                "applied_styles": applied,
                "success": ok_any,
            }
        )
        if ok_any:
            success += 1
        if current != code_val:
            changed += 1
        out_rows.append(out_obj)
        if idx < 20:
            run_log.append(
                {
                    "index": idx,
                    "status": "ok" if ok_any else "no-change",
                    "applied_styles": applied,
                    "changed": current != code_val,
                }
            )

    _write_jsonl(output_path, out_rows)
    preview = out_rows[: min(5, len(out_rows))]
    return TransformDatasetResp(
        total=total, changed=changed, success=success, output_path=str(output_path), preview=preview, log=run_log
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
    return InferTextResp(
        candidates=responses,
        scores=weights,
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
        ],
        decoded=decoded if req.return_decoded else None,
    )


@app.post("/api/infer/dataset", response_model=InferDatasetResp)
def infer_dataset(req: InferDatasetReq) -> InferDatasetResp:
    from ccd.inference.beam_infer import read_jsonl as _read, write_jsonl as _write, extract_response as _extract

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
    for obj in rows:
        code = obj.get(req.field, None)
        if not isinstance(code, str) or not code.strip():
            continue
        # build prompt and generate
        prompt = eng_build_prompt(code, eng_load_system_prompt(pr_cfg))
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():  # type: ignore
            outputs = model.generate(
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
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        seqs = outputs.sequences
        seq_scores = outputs.sequences_scores if hasattr(outputs, "sequences_scores") else None
        weights = None
        if seq_scores is not None:
            weights = eng_softmax(seq_scores).tolist()
        decoded = [tok.decode(s, skip_special_tokens=True) for s in seqs]
        responses = []
        for t in decoded:
            r = _extract(t)
            responses.append(r if r.strip() else (t.strip() or code))

        effective_flat = req.emit_flat or (req.gen.num_return_sequences and req.gen.num_return_sequences > 1)
        if req.write_mode == "generation":
            if effective_flat:
                for i, resp in enumerate(responses):
                    out_obj = dict(obj)
                    out_obj["generation"] = resp
                    out_obj["completion_id"] = i
                    if weights is not None and i < len(weights):
                        out_obj["variant_score"] = weights[i]
                    out_rows.append(out_obj)
            else:
                out_obj = dict(obj)
                out_obj["generation"] = responses[0] if responses else code
                out_rows.append(out_obj)
        else:
            if effective_flat:
                for i, resp in enumerate(responses):
                    out_obj = dict(obj)
                    out_obj[req.field] = resp
                    out_obj["completion_id"] = i
                    if weights is not None and i < len(weights):
                        out_obj["variant_score"] = weights[i]
                    out_rows.append(out_obj)
            else:
                out_obj = dict(obj)
                out_obj[req.field] = responses[0] if responses else code
                out_rows.append(out_obj)

    _write(req.output_path, out_rows)
    ms = int((time.time() - t0) * 1000)
    preview = out_rows[: min(5, len(out_rows))]
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
        ],
    )


