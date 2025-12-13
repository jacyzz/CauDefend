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
from ccd.server.schema import (
    InferDatasetStructuredReq,
    InferDatasetResp,
    InferModelCfg,
    InferGenParams,
    InferPromptCfg,
    InputBuilder,
    OutputSchema
)
from ccd.server.tasks import create_task, update_task, get_task
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

def _worker_dataset_structured(task_id: str, req: InferDatasetStructuredReq):
    try:
        # Late imports to avoid circular dependency
        from ccd.server.main import _get_or_load_model, _read_jsonl, _write_jsonl, _release_model

        update_task(task_id, {"status": "loading_model"})
        
        # 1. Load Model
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

        # 2. Read Input
        rows = _read_jsonl(Path(req.input_path), limit=req.limit or 0)
        total = len(rows)
        update_task(task_id, {"status": "running", "total": total, "current": 0})
        
        out_rows: List[Dict[str, Any]] = []
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
        parser = make_parser(mode="cot" if req.output_schema.extract_sections else "raw", default_analysis=req.output_schema.trace_analysis)
        composer = make_composer(mode=req.output_schema.mode, field=req.output_schema.field, emit_flat=req.output_schema.emit_flat)

        # 4. Loop
        for idx, obj in enumerate(rows):
            # Update progress
            if idx % 5 == 0:
                update_task(task_id, {"current": idx})
            
            input_text = input_builder.build(obj)
            if not input_text:
                continue

            responses, weights, _decoded = generate_for_text(model, tok, input_text, pr_cfg, gen_cfg)

            parsed_candidates = [parser.parse(r) for r in responses]
            out_obj = composer.compose(obj, parsed_candidates)
            
            # Pass-through critical fields if missing
            if "task_id" not in out_obj:
                out_obj["task_id"] = obj.get("task_id")
            if "canonical_solution" not in out_obj and obj.get("canonical_solution") is not None:
                out_obj["canonical_solution"] = obj.get("canonical_solution")
            
            out_rows.append(out_obj)

        # 5. Write Output
        _write_jsonl(Path(req.output_path), out_rows)
        ms = int((time.time() - t0) * 1000)
        preview = out_rows[: min(5, len(out_rows))]
        
        if req.unload_after:
            try:
                _release_model(req.model)
            except Exception:
                pass
        
        result_data = {
            "total": len(out_rows),
            "output_path": req.output_path,
            "preview": preview,
            "elapsed_ms": ms,
        }
        update_task(task_id, {"status": "completed", "current": total, "result": result_data})

    except Exception as e:
        traceback.print_exc()
        update_task(task_id, {"status": "error", "error": str(e)})


@router.post("/api/infer/dataset_structured_async")
def infer_dataset_structured_async(req: InferDatasetStructuredReq):
    task_id = uuid.uuid4().hex
    create_task(task_id)
    t = threading.Thread(target=_worker_dataset_structured, args=(task_id, req), daemon=True)
    t.start()
    return {"task_id": task_id}

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
        "result": info.get("result")
    }
