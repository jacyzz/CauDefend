import threading
import time
import uuid
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query

from ccd.inference.engine import (
    ModelConfig as HFModelConfig,
    PromptConfig as HfPromptConfig,
    GenerationConfig as HfGenConfig,
    generate_for_text,
)
from ccd.inference.remote_openai_compat import OpenAICompatClient
from ccd.server.settings import load_remote_provider_settings
from ccd.server.schema import (
    InferDatasetStructuredReq,
    InferDatasetResp,
    InferModelCfg,
    InferGenParams,
    InferPromptCfg,
    InputBuilder,
    OutputSchema
)
from ccd.server.tasks import create_task, update_task, get_task, list_tasks, request_cancel
from ccd.inference.processors import (
    MergeFieldsBuilder,
    make_parser,
    make_composer,
)

# Need to import helper functions from main or move them to util.
# Moving helpers to util would be cleaner, but for now we import from main with caution or replicate.
# Replicating small helpers to avoid circular import if main imports api_async.
# Actually, main imports api_async, so api_async CANNOT import main if main also imports api_async.
# So we must move shared logic to a separate module.

# Let's use a dynamic import inside functions or move helpers to ccd.server.utils
# Dynamic import is easiest for now.

router = APIRouter()

_LOCAL_DATASET_STRUCTURED_SEM = threading.Semaphore(1)


def _is_cancelled(task_id: str) -> bool:
    info = get_task(task_id)
    return bool(info.get("cancel_requested"))

def _worker_dataset_structured(task_id: str, req: InferDatasetStructuredReq):
    remote_client = None
    provider = (getattr(req, "provider", "local") or "local").strip() or "local"
    local_slot_acquired = False
    try:
        # Late imports to avoid circular dependency
        from ccd.server.main import _get_or_load_model, _release_model
        from ccd.server.jsonl_io import JsonlAtomicWriter, iter_jsonl, count_jsonl

        if provider == "local":
            update_task(task_id, {"status": "queued"})
            _LOCAL_DATASET_STRUCTURED_SEM.acquire()
            local_slot_acquired = True

        update_task(task_id, {"status": "loading_model"})

        # 1. Load Model (or prepare remote client)
        model = None
        tok = None
        if provider == "local":
            model, tok = _get_or_load_model(req.model)
        else:
            try:
                settings = load_remote_provider_settings(provider)
            except ValueError as e:
                update_task(task_id, {"status": "error", "error": str(e)[:800]})
                return
            remote_client = OpenAICompatClient(settings)
            remote_client.__enter__()
        
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

        if _is_cancelled(task_id):
            update_task(task_id, {"status": "cancelled", "error": "cancelled"})
            return

        # 2. Count Input (for progress)
        total = count_jsonl(Path(req.input_path), limit=req.limit or 0)
        update_task(task_id, {"status": "running", "total": total, "current": 0})

        out_total = 0
        preview: List[Dict[str, Any]] = []
        t0 = time.time()

        # 3. Setup Processors
        ib_cfg = req.input_builder
        if ib_cfg.mode == "single":
            fields = [ib_cfg.field] if ib_cfg.field else []
        else:
            fields = ib_cfg.fields if (isinstance(ib_cfg.fields, list) and len(ib_cfg.fields) > 0) else ["declaration", "canonical_solution"]
        
        input_builder = MergeFieldsBuilder(
            fields=fields,
            separator=ib_cfg.separator or "\n\n",
            prefix=ib_cfg.prefix or "",
            suffix=ib_cfg.suffix or "",
        )
        parser = make_parser(
            mode="cot" if req.output_schema.extract_sections else "raw",
            default_analysis=req.output_schema.trace_analysis,
        )
        composer = make_composer(
            mode=req.output_schema.mode,
            field=req.output_schema.field,
            emit_flat=req.output_schema.emit_flat,
            keep_original_fields=req.output_schema.keep_original_fields,
        )

        # 4. Loop + stream write
        with JsonlAtomicWriter(Path(req.output_path)) as writer:
            for idx, obj in enumerate(iter_jsonl(Path(req.input_path), limit=req.limit or 0)):
                if _is_cancelled(task_id):
                    # leave partial output file for inspection
                    update_task(task_id, {"status": "cancelled", "current": idx})
                    return
                # Update progress
                if idx % 5 == 0:
                    update_task(task_id, {"current": idx})

                input_text = input_builder.build(obj)
                if not input_text:
                    continue

                if provider == "local":
                    responses, _weights, _decoded = generate_for_text(model, tok, input_text, pr_cfg, gen_cfg)
                else:
                    assert remote_client is not None
                    result = remote_client.generate(
                        model=req.model.model,
                        input_text=input_text,
                        prompt_cfg=pr_cfg,
                        gen_cfg=gen_cfg,
                    )
                    responses, _weights, _decoded = result.candidates, None, result.decoded

                parsed_candidates = [parser.parse(r) for r in responses]
                out_obj = composer.compose(obj, parsed_candidates)
                writer.write_obj(out_obj)
                out_total += 1
                if len(preview) < 5:
                    preview.append(out_obj)
        ms = int((time.time() - t0) * 1000)
        
        if req.unload_after:
            try:
                if provider == "local":
                    _release_model(req.model)
            except Exception:
                pass
        
        result_data = {
            "total": out_total,
            "output_path": req.output_path,
            "preview": preview,
            "elapsed_ms": ms,
        }
        update_task(task_id, {"status": "completed", "current": total, "result": result_data})

    except Exception as e:
        traceback.print_exc()
        err = str(e)
        if len(err) > 800:
            err = err[:800] + "..."
        update_task(task_id, {"status": "error", "error": err})
    finally:
        if remote_client is not None:
            try:
                remote_client.__exit__(None, None, None)
            except Exception:
                pass
        if local_slot_acquired:
            try:
                _LOCAL_DATASET_STRUCTURED_SEM.release()
            except Exception:
                pass


@router.post("/api/infer/dataset_structured_async")
def infer_dataset_structured_async(req: InferDatasetStructuredReq):
    task_id = uuid.uuid4().hex
    provider = (getattr(req, "provider", "local") or "local").strip() or "local"
    model_name = getattr(getattr(req, "model", None), "model", "") or ""
    create_task(
        task_id,
        kind="infer_dataset_structured",
        provider=provider,
        input_path=req.input_path,
        output_path=req.output_path,
        model=model_name,
    )
    t = threading.Thread(target=_worker_dataset_structured, args=(task_id, req), daemon=True)
    t.start()
    return {"task_id": task_id}


@router.get("/api/infer/tasks")
def infer_list_tasks() -> Dict[str, Any]:
    tasks = list_tasks(kind="infer_dataset_structured")
    # Keep payload small; frontend can query /progress if it needs full result
    summaries: List[Dict[str, Any]] = []
    for t in tasks:
        tid = t.get("task_id")
        if not tid:
            continue
        cur = int(t.get("current", 0) or 0)
        tot = int(t.get("total", 0) or 0)
        pct = (cur / tot * 100.0) if tot > 0 else 0.0
        summaries.append(
            {
                "task_id": tid,
                "kind": t.get("kind"),
                "provider": t.get("provider"),
                "model": t.get("model"),
                "input_path": t.get("input_path"),
                "output_path": t.get("output_path"),
                "status": t.get("status"),
                "current": cur,
                "total": tot,
                "percent": pct,
                "error": t.get("error"),
                "cancel_requested": bool(t.get("cancel_requested")),
                "created_at": t.get("created_at"),
                "updated_at": t.get("updated_at"),
                # result is optional and can be large; keep only output_path/elapsed_ms when present
                "result": t.get("result"),
            }
        )
    return {"tasks": summaries}


@router.post("/api/infer/tasks/{task_id}/cancel")
def infer_cancel_task(task_id: str) -> Dict[str, Any]:
    ok = request_cancel(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Task not found")
    update_task(task_id, {"status": "cancelling"})
    return {"ok": True}

@router.get("/api/infer/progress")
def infer_progress(task_id: str = Query(...)):
    info = get_task(task_id)
    if not info:
        raise HTTPException(status_code=404, detail="Task not found")
    
    current = info.get("current", 0)
    total = info.get("total", 1)
    percent = 0.0
    if total > 0:
        percent = (current / total) * 100.0
        
    return {
        "task_id": task_id,
        "status": info.get("status"),
        "current": current,
        "total": total,
        "percent": percent,
        "error": info.get("error"),
        "result": info.get("result"),
        "cancel_requested": bool(info.get("cancel_requested")),
        "provider": info.get("provider"),
        "model": info.get("model"),
        "input_path": info.get("input_path"),
        "output_path": info.get("output_path"),
        "created_at": info.get("created_at"),
        "updated_at": info.get("updated_at"),
    }
