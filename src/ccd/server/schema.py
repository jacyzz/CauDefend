"""
Shared Pydantic models for API
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# --- IST ---

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

# --- Inference Common ---

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

# --- Inference Text ---

class InferTextReq(BaseModel):
    provider: str = Field("local", description="'local' for HF; otherwise uses OpenAI-compatible remote provider")
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
    structured_candidates: Optional[List[Dict[str, str]]] = None

# --- Inference Dataset (Classic) ---

class InferDatasetReq(BaseModel):
    provider: str = Field("local", description="'local' for HF; otherwise uses OpenAI-compatible remote provider")
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

# --- Inference Dataset (Structured) ---

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
    keep_original_fields: bool = True  # keep all original JSONL fields in output rows


class InferDatasetStructuredReq(BaseModel):
    provider: str = Field("local", description="'local' for HF; otherwise uses OpenAI-compatible remote provider")
    input_path: str
    output_path: str
    input_builder: InputBuilder = InputBuilder()
    output_schema: OutputSchema = OutputSchema()
    model: InferModelCfg
    prompt: InferPromptCfg = InferPromptCfg()
    gen: InferGenParams = InferGenParams()
    limit: int = 0
    unload_after: bool = False

# --- DSPy ---

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
    custom_prompt_text: Optional[str] = None
    custom_vars: Optional[Dict[str, Any]] = None
    extract_code: bool = False
    combine_fields: bool = False
    prompt_field: Optional[str] = None
    output_prompt_field: Optional[str] = None
    output_code_field: Optional[str] = None


class DSpyDatasetResp(BaseModel):
    total: int
    output_path: str
    elapsed_ms: int
    preview: List[Dict[str, Any]] = []

# --- Tasks ---

class AsyncProgress(BaseModel):
    task_id: str
    status: str = "pending"  # pending, running, completed, error
    current: int = 0
    total: int = 0
    percent: float = 0.0
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class InferUnloadReq(BaseModel):
    model: InferModelCfg
