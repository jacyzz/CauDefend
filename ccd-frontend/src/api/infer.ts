import { http } from './http';

export type InferModelCfg = {
  model: string;
  dtype?: 'float16' | 'bfloat16' | 'auto';
  device_map?: string;
  trust_remote_code?: boolean;
  low_cpu_mem_usage?: boolean;
  use_safetensors?: boolean;
  base_model?: string;
  peft_adapter?: string;
  peft_merge?: boolean;
};

export type InferPromptCfg = {
  template_yaml?: string;
  system_prompt_text?: string;
};

export type InferGenParams = {
  max_new_tokens?: number;
  do_sample?: boolean;
  temperature?: number;
  top_p?: number;
  num_beams?: number;
  num_return_sequences?: number;
  num_beam_groups?: number;
  diversity_penalty?: number;
  seed?: number;
};

export type InferTextReq = {
  provider?: string;
  input_text: string;
  model: InferModelCfg;
  prompt?: InferPromptCfg;
  gen?: InferGenParams;
  return_decoded?: boolean;
  unload_after?: boolean;
};

export type InferTextResp = {
  candidates: string[];
  scores?: number[];
  elapsed_ms: number;
  log: string[];
  decoded?: string[];
  structured_candidates?: { analysis: string; code: string }[];
};

export async function inferGenerate(body: InferTextReq) {
  const { data } = await http.post<InferTextResp>('/infer/generate', body);
  return data;
}

export type InferDatasetReq = {
  provider?: string;
  input_path: string;
  output_path: string;
  field: string;
  output_field?: string;
  model: InferModelCfg;
  prompt?: InferPromptCfg;
  gen?: InferGenParams;
  emit_flat?: boolean;
  write_mode?: 'generation' | 'overwrite';
  limit?: number;
  unload_after?: boolean;
  // optional: merge prompt+code into one input, then split output back
  combine_fields?: boolean;
  prompt_field?: string;
  output_prompt_field?: string;
  output_code_field?: string;
  extract_code?: boolean;
};

export type InferDatasetResp = {
  total: number;
  output_path: string;
  preview: any[];
  elapsed_ms: number;
  log: string[];
};

export async function inferDataset(body: InferDatasetReq) {
  const { data } = await http.post<InferDatasetResp>('/infer/dataset', body);
  return data;
}

// ---------- Config-driven structured dataset ----------
export type InputBuilder = {
  mode?: 'merge' | 'single';
  field?: string;
  fields?: string[];
  separator?: string;
  prefix?: string;
  suffix?: string;
  id_field?: string;
};

export type OutputSchema = {
  mode?: 'structured_variants' | 'single_field';
  field?: string;
  emit_flat?: boolean;
  preset?: 'humaneval_structured' | string;
  trace_analysis?: string;
  extract_sections?: boolean;
  keep_original_fields?: boolean;
};

export type InferDatasetStructuredReq = {
  provider?: string;
  input_path: string;
  output_path: string;
  input_builder?: InputBuilder;
  output_schema?: OutputSchema;
  model: InferModelCfg;
  prompt?: InferPromptCfg;
  gen?: InferGenParams;
  limit?: number;
  unload_after?: boolean;
};

export async function inferDatasetStructured(body: InferDatasetStructuredReq) {
  const { data } = await http.post<InferDatasetResp>('/infer/dataset_structured', body);
  return data;
}

export async function inferDatasetStructuredAsync(body: InferDatasetStructuredReq) {
  const { data } = await http.post<{ task_id: string }>('/infer/dataset_structured_async', body);
  return data;
}

export type InferProgressResp = {
  task_id: string;
  status: 'pending' | 'queued' | 'loading_model' | 'running' | 'cancelling' | 'cancelled' | 'completed' | 'error';
  current: number;
  total: number;
  percent: number;
  error?: string;
  result?: InferDatasetResp;
  cancel_requested?: boolean;
  provider?: string;
  model?: string;
  input_path?: string;
  output_path?: string;
  created_at?: number;
  updated_at?: number;
};

export async function getInferProgress(task_id: string) {
  const { data } = await http.get<InferProgressResp>('/infer/progress', { params: { task_id } });
  return data;
}

export type InferTaskSummary = InferProgressResp & {
  kind?: string;
};

export async function listInferTasks() {
  const { data } = await http.get<{ tasks: InferTaskSummary[] }>('/infer/tasks');
  return data;
}

export async function cancelInferTask(task_id: string) {
  const { data } = await http.post<{ ok: boolean }>(`/infer/tasks/${encodeURIComponent(task_id)}/cancel`, {});
  return data;
}

export async function inspectJsonl(path: string, limit = 5) {
  const { data } = await http.get<{ path: string; count_preview: number; fields: string[]; preview: any[] }>(
    '/infer/inspect_jsonl',
    { params: { path, limit } },
  );
  return data;
}

export async function unloadModel(body: { model: InferModelCfg }): Promise<{ ok: boolean }> {
  const { data } = await http.post<{ ok: boolean }>('/infer/unload', body);
  return data;
}

export async function unloadAll(): Promise<{ ok: boolean; cleared: boolean; remaining: number }> {
  const { data } = await http.post<{ ok: boolean; cleared: boolean; remaining: number }>('/infer/unload_all', {});
  return data;
}

// ---------- DSPy endpoints ----------
export type DSpyModelCfg = {
  model: string;
  dtype?: 'float16' | 'bfloat16' | 'auto';
  device_map?: string;
};

export type DSpyGenParams = {
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  do_sample?: boolean;
  n?: number;
  num_beams?: number;
  early_stopping?: boolean;
};

export type DSpyTextReq = {
  input_text: string;
  signature_mode?: 'completion' | 'defense' | 'custom' | 'freeform';
  model: DSpyModelCfg;
  gen?: DSpyGenParams;
  unload_after?: boolean;
  custom_prompt_text?: string;
  custom_vars?: Record<string, any>;
  extract_code?: boolean;
};

export type DSpyTextResp = {
  candidates: string[];
  analyses?: string[];
  elapsed_ms: number;
};

export async function dspyGenerate(body: DSpyTextReq) {
  const { data } = await http.post<DSpyTextResp>('/dspy/generate', body);
  return data;
}

export type DSpyDatasetReq = {
  input_path: string;
  output_path: string;
  field: string;
  signature_mode?: 'completion' | 'defense' | 'custom' | 'freeform';
  model: DSpyModelCfg;
  gen?: DSpyGenParams;
  emit_flat?: boolean;
  write_mode?: 'generation' | 'overwrite';
  output_field?: string;
  limit?: number;
  unload_after?: boolean;
  custom_prompt_text?: string;
  custom_vars?: Record<string, any>;
  extract_code?: boolean;
  combine_fields?: boolean;
  prompt_field?: string;
  output_prompt_field?: string;
  output_code_field?: string;
};

export type DSpyDatasetResp = {
  total: number;
  output_path: string;
  elapsed_ms: number;
  preview: any[];
};

export async function dspyDataset(body: DSpyDatasetReq) {
  const { data } = await http.post<DSpyDatasetResp>('/dspy/dataset', body);
  return data;
}
