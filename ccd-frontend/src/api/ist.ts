import { http } from './http';

export type StyleItem = {
  code: string;
  family: string;
  type: string;
  subtype: string;
  prepare?: string | null;
};

export async function fetchStyles() {
  const { data } = await http.get<{ styles: StyleItem[] }>('/ist/styles');
  return data.styles;
}

export type TransformTextReq = {
  language: string;
  code: string;
  strategy: 'fixed' | 'random';
  styles?: string[];
  poison_min?: number;
  poison_max?: number;
  avoid_similar?: boolean;
  seed?: number;
};

export type TransformTextResp = {
  converted_code: string;
  applied_styles: string[];
  syntax_ok: boolean;
  processing_time_ms: number;
  log: string[];
};

export async function transformText(body: TransformTextReq) {
  const { data } = await http.post<TransformTextResp>('/ist/transform_text', body);
  return data;
}

export type TransformDatasetReq = {
  input_path: string;
  output_path: string;
  language: string;
  code_field: string;
  combine_fields?: boolean;
  prompt_field?: string;
  output_prompt_field?: string;
  output_code_field?: string;
  id_field?: string;
  backup_field?: string;
  strategy: 'fixed' | 'random';
  styles?: string[];
  poison_candidates?: string[];
  poison_min?: number;
  poison_max?: number;
  avoid_similar?: boolean;
  limit?: number;
  seed?: number;
  syntax_check?: boolean;
};

export type TransformDatasetResp = {
  total: number;
  changed: number;
  success: number;
  output_path: string;
  preview: any[];
  log: { index: number; status: string; [k: string]: any }[];
};

export async function transformDataset(body: TransformDatasetReq) {
  const { data } = await http.post<TransformDatasetResp>('/ist/transform_dataset', body);
  return data;
}

// Async transform with progress polling
export async function transformDatasetAsync(body: TransformDatasetReq) {
  const { data } = await http.post<{ task_id: string; total: number; status: string }>('/ist/transform_dataset_async', body);
  return data;
}

export async function fetchTransformProgress(task_id: string) {
  const { data } = await http.get('/ist/progress', { params: { task_id } });
  return data as {
    task_id: string;
    status: 'running' | 'done' | 'error';
    current: number;
    total: number;
    percent: number;
    error?: string;
    result?: TransformDatasetResp;
  };
}

export async function fetchDatasetSchema(path: string, preview = 5) {
  const { data } = await http.get<{ path: string; count_preview: number; fields: string[]; preview: any[] }>(
    '/ist/dataset_schema',
    { params: { path, preview } },
  );
  return data;
}

export async function fetchRecord(params: { path: string; index: number; code_field: string; backup_field?: string }) {
  const { data } = await http.get('/ist/record', { params });
  return data as { index: number; record: any; before?: string; after?: string };
}


