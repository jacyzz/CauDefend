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
};

export async function inferGenerate(body: InferTextReq) {
  const { data } = await http.post<InferTextResp>('/infer/generate', body);
  return data;
}

export type InferDatasetReq = {
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

export async function unloadModel(body: { model: InferModelCfg }): Promise<{ ok: boolean }> {
  const { data } = await http.post<{ ok: boolean }>('/infer/unload', body);
  return data;
}

export async function unloadAll(): Promise<{ ok: boolean; cleared: boolean; remaining: number }> {
  const { data } = await http.post<{ ok: boolean; cleared: boolean; remaining: number }>('/infer/unload_all', {});
  return data;
}


