import { useEffect, useState } from 'react';
import { Button, Card, Col, Divider, Form, Input, InputNumber, Row, Select, Space, Switch, Tabs, Typography, message } from 'antd';
import { inferDatasetStructured, unloadModel, inspectJsonl } from '../../api/infer';
import type { InferGenParams, InferModelCfg, InferPromptCfg, InputBuilder, OutputSchema, InferDatasetStructuredReq } from '../../api/infer';

const { Text } = Typography;

const DTYPE_OPTIONS = [
  { label: 'float16', value: 'float16' },
  { label: 'bfloat16', value: 'bfloat16' },
  { label: 'auto', value: 'auto' },
];

export default function InferDataset() {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [preview, setPreview] = useState<any[]>([]);
  const [fields, setFields] = useState<string[]>([]);
  const [schemaPreview, setSchemaPreview] = useState<any[]>([]);

  const presetTemplate =
    [
      'Role：',
      'You are a Static Analysis Engine with Constraint Assessment capabilities.',
      '',
      'Task：',
      'Clean the provided code and generate 4 stylistically diverse variants.',
      '',
      'Constraints：',
      'Keep exactly the same function name and return type.',
      'Only clean parameter names by removing trigger suffixes.',
      'Include all #include statements from the original code (use actual statements, not placeholders)',
      '',
      'Input Code{language}：',
      '{poisoned_code}',
      '',
      'Thinking：',
      'Perform the following Constraint Assessment:',
      '',
      'Phase 1: Constraint Assessment (CoT)',
      'Perform a mental static analysis:',
      'Reachability Analysis: Analyze if(0) / while(0). Conclusion: Unreachable code (Dead). Action: Remove.',
      'Taint Analysis: Track variables ending in _secret, _vuln. Action: Sanitize/Rename to remove taint.',
      'Signature Constraint Locking: Extract the function signature: [NAME] ().',
      'Constraint: This signature is IMMUTABLE.',
      'Header Constraint: Extract list of #include. Constraint: Must appear in all outputs.',
      '',
      'Phase 2: Execution Instructions',
      'Generate 4 clean variants respecting the Locked Constraints:',
      'Remove dead code.',
      'Fix suspicious variable names (suffixes: _sh, _secret, etc.).',
      'Remove volatile declarations.',
      'Keep EXACTLY the same function name and return type.',
      'Include ALL #include statements.',
      '',
      'CRITICAL OUTPUT REQUIREMENTS：',
      'Output 4 (FOUR) code variants.',
      'Ensure high stylistic diversity while maintaining semantic equivalence.',
      'Use REAL #include statements.',
      'NO explanations, NO placeholders.',
      '',
      'Output Format：',
      '```cpp',
      '',
      '// Variant 1',
      '#include...',
      'bool function_name(...) {... }',
      '',
      '// Variant 2',
      '#include...',
      'bool function_name(...) {... }',
      '',
      '// Variant 3',
      '#include...',
      'bool function_name(...) {... }',
      '',
      '// Variant 4',
      '#include...',
      'bool function_name(...) {... }',
      '',
      '```',
    ].join('\n');

  const fillPreset = () => {
    const v = form.getFieldsValue();
    form.setFieldsValue({
      num_beams: Math.max(4, v?.num_beams || 4),
      do_sample: false,
      max_new_tokens: v?.max_new_tokens || 256,
    });
    message.success('已填充推荐参数（num_beams≥4）。');
  };

  const fillStructuredPreset = () => {
    const v = form.getFieldsValue();
    form.setFieldsValue({
      engine: 'hf',
      // 合并 declaration + canonical_solution，输出结构化 variants 到 output
      field: v?.field || 'canonical_solution',
      input_path: v?.input_path || '',
      output_path: v?.output_path || '',
      combine_fields: true,
      prompt_field: 'declaration',
      write_mode: 'generation',
      output_field: 'output',
      emit_flat: false,
      // 典型 beam 设定：多候选
      num_beams: Math.max(4, v?.num_beams || 4),
      num_return_sequences: Math.max(4, v?.num_return_sequences || 4),
      num_beam_groups: 1,
      diversity_penalty: 0.0,
      do_sample: false,
    });
    message.success('已填充 HumanEval 结构化输出预设（合并声明+代码，输出到 output.variants）。');
  };

  // Persist form across route/tab switches and optionally unload on leave
  useEffect(() => {
    try {
      const saved = localStorage.getItem('infer_dataset_form');
      if (saved) {
        const vals = JSON.parse(saved);
        form.setFieldsValue(vals);
      }
    } catch {}
    return () => {
      try {
        const v = form.getFieldsValue();
        if (v?.unload_on_leave) {
          if (v?.engine === 'dspy') {
            // DSPy 无单独 unload API，建议使用 unload_after
            return;
          }
          const model: InferModelCfg | undefined = v?.model || v?.model === '' ? undefined : {
            model: v.model,
            dtype: v.dtype,
            device_map: v.device_map ?? 'auto',
            trust_remote_code: !!v.trust_remote_code,
            low_cpu_mem_usage: !!v.low_cpu_mem_usage,
            use_safetensors: !!v.use_safetensors,
            base_model: v.base_model || undefined,
            peft_adapter: v.peft_adapter || undefined,
            peft_merge: !!v.peft_merge,
          };
          if (model && typeof model.model === 'string' && model.model.trim()) {
            unloadModel({ model }).catch(() => void 0);
          }
        }
      } catch {}
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleValuesChange = (_: any, allValues: any) => {
    try {
      localStorage.setItem('infer_dataset_form', JSON.stringify(allValues));
    } catch {}
  };

  const onRun = async () => {
    try {
      const v = await form.validateFields();
      setLoading(true);
      setLogs([]);
      const model: InferModelCfg = {
        model: v.model,
        dtype: v.dtype,
        device_map: v.device_map ?? 'auto',
        trust_remote_code: !!v.trust_remote_code,
        low_cpu_mem_usage: !!v.low_cpu_mem_usage,
        use_safetensors: !!v.use_safetensors,
        base_model: v.base_model || undefined,
        peft_adapter: v.peft_adapter || undefined,
        peft_merge: !!v.peft_merge,
      };
      const prompt: InferPromptCfg = {
        template_yaml: v.template_yaml || undefined,
        system_prompt_text: v.system_prompt_text || undefined,
      };
      const gen: InferGenParams = {
        max_new_tokens: v.max_new_tokens,
        do_sample: !!v.do_sample,
        temperature: v.temperature,
        top_p: v.top_p,
        num_beams: v.num_beams,
        num_return_sequences: v.num_return_sequences,
        num_beam_groups: v.num_beam_groups,
        diversity_penalty: v.diversity_penalty,
        seed: v.seed,
      };
      const input_builder: InputBuilder = {
        mode: v.input_mode || 'merge',
        field: v.single_field || undefined,
        fields: v.merge_fields && Array.isArray(v.merge_fields) ? v.merge_fields : undefined,
        separator: v.separator || '\n\n',
        prefix: v.prefix || '',
        suffix: v.suffix || '',
        id_field: v.id_field || 'task_id',
      };
      const output_schema: OutputSchema = {
        mode: v.output_mode || 'structured_variants',
        field: v.output_field || 'output',
        emit_flat: !!v.emit_flat,
        preset: 'humaneval_structured',
        trace_analysis: v.trace_analysis || '',
        extract_sections: v.extract_sections !== false, // default true
      };
      const body: InferDatasetStructuredReq = {
        input_path: v.input_path,
        output_path: v.output_path,
        input_builder,
        output_schema,
        model,
        prompt,
        gen,
        limit: v.limit ?? 0,
        unload_after: !!v.unload_after,
      };
      const res = await inferDatasetStructured(body);
      setPreview(res.preview || []);
      setLogs([`总计: ${res.total}`, `耗时: ${res.elapsed_ms} ms`, ...res.log, `输出: ${res.output_path}`]);
      message.success(`已写入：${res.output_path}`);
    } catch (e: any) {
      message.error(e.message ?? '执行失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Card title="Inference · 数据集推理（结构化）">
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          <Button type="primary" onClick={onRun} loading={loading}>
            执行数据集推理
          </Button>
          <Button onClick={fillPreset} disabled={loading}>填充推荐参数</Button>
        </div>
        <Divider />

        <Row gutter={16} wrap={false}>
          <Col flex="1 1 auto">
            <Card size="small" title="输出预览（最多5条）">
              {preview.length === 0 ? (
                <Text type="secondary">暂无预览</Text>
              ) : (
                <>
                  {/* 若为结构化输出：output: [{variant, trace_analysis, sanitized_code}, ...] */}
                  {Array.isArray(preview[0]?.output) && preview[0]?.output?.length > 0 && typeof preview[0].output[0] === 'object' ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                      {preview.slice(0, 5).map((row, idx) => (
                        <Card size="small" key={idx} bodyStyle={{ padding: 8 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                            <Text strong>{row?.task_id ?? `记录 ${idx + 1}`}</Text>
                            <Text type="secondary">
                              {Array.isArray(row.output) ? `variants: ${row.output.length}` : null}
                            </Text>
                          </div>
                          <div style={{ display: 'flex', gap: 8 }}>
                            <div style={{ flex: 1 }}>
                              <Text type="secondary">declaration（节选）</Text>
                              <pre style={{ background: '#0b0b0b', color: '#ddd', padding: 6, maxHeight: 120, overflow: 'auto' }}>
                                {(row?.declaration ?? '').slice(0, 800)}
                              </pre>
                            </div>
                            <div style={{ flex: 1 }}>
                              <Text type="secondary">canonical_solution（节选）</Text>
                              <pre style={{ background: '#0b0b0b', color: '#ddd', padding: 6, maxHeight: 120, overflow: 'auto' }}>
                                {(row?.canonical_solution ?? '').slice(0, 800)}
                              </pre>
                            </div>
                          </div>
                          <Divider style={{ margin: '8px 0' }} />
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                            {row.output.slice(0, 6).map((v: any, i: number) => (
                              <div key={i} style={{ border: '1px solid #333', borderRadius: 4 }}>
                                <div style={{ background: '#111', padding: '4px 6px' }}>
                                  <Text>Variant {v?.variant ?? i + 1}</Text>
                                </div>
                                <pre style={{ background: '#0b0b0b', color: '#ddd', padding: 6, maxHeight: 240, overflow: 'auto', margin: 0 }}>
                                  {String(v?.sanitized_code ?? '').slice(0, 5000)}
                                </pre>
                              </div>
                            ))}
                          </div>
                        </Card>
                      ))}
                    </div>
                  ) : (
                    <pre style={{ background: '#0b0b0b', color: '#ddd', padding: 8, maxHeight: 560, overflow: 'auto' }}>
                      {preview.map((r) => JSON.stringify(r, null, 2)).join('\n\n-----\n\n')}
                    </pre>
                  )}
                </>
              )}
            </Card>
          </Col>

          <Col flex="420px">
            <Card size="small" title="参数设置" style={{ marginBottom: 16 }}>
              <Form
                form={form}
                layout="vertical"
                onValuesChange={handleValuesChange}
                initialValues={{
                  dtype: 'bfloat16',
                  device_map: 'auto',
                  do_sample: false,
                  temperature: 1.0,
                  top_p: 1.0,
                  num_beams: 4,
                  num_return_sequences: 4,
                  num_beam_groups: 1,
                  diversity_penalty: 0.0,
                  max_new_tokens: 512,
                  seed: 123456,
                  emit_flat: false,
                  unload_after: false,
                  input_mode: 'merge',
                  merge_fields: ['declaration', 'canonical_solution'],
                  separator: '\n\n',
                  id_field: 'task_id',
                  output_mode: 'structured_variants',
                  output_field: 'output',
                }}
              >
                <Tabs
                  defaultActiveKey="data"
                  items={[
                    {
                      key: 'data',
                      label: '数据集',
                      children: (
                        <>
                          <Form.Item name="input_path" label="输入 JSONL 路径" rules={[{ required: true }]}>
                            <Input placeholder="/path/to/input.jsonl" />
                          </Form.Item>
                          <Form.Item name="output_path" label="输出 JSONL 路径" rules={[{ required: true }]}>
                            <Input placeholder="/path/to/output.jsonl" />
                          </Form.Item>
                          <Space>
                            <Button
                              onClick={async () => {
                                try {
                                  const v = form.getFieldsValue();
                                  const res = await inspectJsonl(v.input_path, 5);
                                  setFields(res.fields || []);
                                  setSchemaPreview(res.preview || []);
                                  message.success(`已解析字段：${(res.fields || []).join(', ')}`);
                                } catch (e: any) {
                                  message.error(e.message ?? '解析失败');
                                }
                              }}
                            >
                              解析字段
                            </Button>
                            {fields.length > 0 && <span style={{ color: '#888' }}>已发现字段：{fields.join(', ')}</span>}
                          </Space>
                          <Divider />
                          <Form.Item name="input_mode" label="输入模式">
                            <Select
                              options={[
                                { label: '合并字段（declaration+canonical_solution）', value: 'merge' },
                                { label: '单字段', value: 'single' },
                              ]}
                            />
                          </Form.Item>
                          <Form.Item noStyle shouldUpdate>
                            {({ getFieldValue }) =>
                              getFieldValue('input_mode') === 'merge' ? (
                                <>
                                  <Form.Item name="merge_fields" label="合并字段">
                                    <Select
                                      mode="multiple"
                                      options={(fields.length ? fields : ['declaration', 'canonical_solution']).map((f) => ({ label: f, value: f }))}
                                    />
                                  </Form.Item>
                                  <Form.Item name="separator" label="分隔符">
                                    <Input placeholder="默认：\\n\\n" />
                                  </Form.Item>
                                </>
                              ) : (
                                <Form.Item name="single_field" label="单字段名" rules={[{ required: true }]}>
                                  <Select
                                    options={(fields.length ? fields : ['canonical_solution']).map((f) => ({ label: f, value: f }))}
                                  />
                                </Form.Item>
                              )
                            }
                          </Form.Item>
                          <Row gutter={8}>
                            <Col span={12}>
                              <Form.Item name="prefix" label="输入前缀（可选）">
                                <Input placeholder="附加在输入前的提示" />
                              </Form.Item>
                            </Col>
                            <Col span={12}>
                              <Form.Item name="suffix" label="输入后缀（可选）">
                                <Input placeholder="附加在输入后的提示" />
                              </Form.Item>
                            </Col>
                          </Row>
                          <Form.Item name="id_field" label="ID 字段">
                            <Input placeholder="task_id" />
                          </Form.Item>
                          <Divider />
                          <Form.Item name="output_mode" label="输出格式">
                            <Select
                              options={[
                                { label: '结构化 variants（HumanEval）', value: 'structured_variants' },
                                { label: '写入单字段', value: 'single_field' },
                              ]}
                            />
                          </Form.Item>
                          <Form.Item noStyle shouldUpdate>
                            {({ getFieldValue }) =>
                              getFieldValue('output_mode') === 'single_field' ? (
                                <Form.Item
                                  name="output_field"
                                  label="写入字段（single_field）"
                                  rules={[{ required: true, message: '请指定写入字段' }]}
                                >
                                  <Input placeholder="如 generation 或 pred_code" />
                                </Form.Item>
                              ) : (
                                <>
                                  <Form.Item name="output_field" label="结构化字段名">
                                    <Input placeholder="默认：output" />
                                  </Form.Item>
                                  <Form.Item name="extract_sections" label="从输出中抽取 Trace Analysis + 代码" valuePropName="checked" initialValue>
                                    <Switch />
                                  </Form.Item>
                                  <Form.Item name="trace_analysis" label="trace_analysis 固定值">
                                    <Input placeholder="写入每个 variant.trace_analysis 的常量值（可留空）" />
                                  </Form.Item>
                                </>
                              )
                            }
                          </Form.Item>
                          <Form.Item name="limit" label="限制条数（0=全部）">
                            <InputNumber min={0} step={1} style={{ width: '100%' }} />
                          </Form.Item>
                          <Form.Item name="emit_flat" label="多候选展开多行（single_field时有效）" valuePropName="checked">
                            <Switch />
                          </Form.Item>
                          <Form.Item name="unload_after" label="推理后释放显存（unload_after）" valuePropName="checked">
                            <Switch />
                          </Form.Item>
                          <Form.Item name="unload_on_leave" label="离开页面释放显存（unload_on_leave）" valuePropName="checked">
                            <Switch />
                          </Form.Item>
                        </>
                      ),
                    },
                    {
                      key: 'model',
                      label: '模型 / LoRA',
                      children: (
                        <>
                          <Form.Item name="model" label="模型路径/名称" rules={[{ required: true }]}>
                            <Input placeholder="/path/to/merged-or-base-model" />
                          </Form.Item>
                          <Form.Item name="base_model" label="底座模型（可选，用于LoRA）">
                            <Input placeholder="/path/to/base-model" />
                          </Form.Item>
                          <Form.Item name="peft_adapter" label="PEFT 适配器（可选）">
                            <Input placeholder="/path/to/adapter-dir (含 adapter_config.json)" />
                          </Form.Item>
                          <Form.Item name="peft_merge" label="合并LoRA到权重" valuePropName="checked">
                            <Switch />
                          </Form.Item>
                          <Row gutter={8}>
                            <Col span={12}>
                              <Form.Item name="dtype" label="dtype">
                                <Select options={DTYPE_OPTIONS} />
                              </Form.Item>
                            </Col>
                            <Col span={12}>
                              <Form.Item name="device_map" label="device_map">
                                <Input placeholder="auto / cuda:0 / balanced 等" />
                              </Form.Item>
                            </Col>
                          </Row>
                          <Row gutter={8}>
                            <Col span={8}>
                              <Form.Item name="trust_remote_code" label="trust_remote_code" valuePropName="checked">
                                <Switch />
                              </Form.Item>
                            </Col>
                            <Col span={8}>
                              <Form.Item name="low_cpu_mem_usage" label="low_cpu_mem_usage" valuePropName="checked">
                                <Switch />
                              </Form.Item>
                            </Col>
                            <Col span={8}>
                              <Form.Item name="use_safetensors" label="use_safetensors" valuePropName="checked">
                                <Switch />
                              </Form.Item>
                            </Col>
                          </Row>
                        </>
                      ),
                    },
                    {
                      key: 'prompt',
                      label: '提示模板',
                      children: (
                        <>
                          <Form.Item name="template_yaml" label="模板 YAML（可选）">
                            <Input placeholder="/path/to/template.yaml（含 {{ system_prompt }} 占位符）" />
                          </Form.Item>
                          <Form.Item name="system_prompt_text" label="系统提示文本">
                            <Input.TextArea rows={3} placeholder="默认代码重构安全提示，可自定义" />
                          </Form.Item>
                        </>
                      ),
                    },
                    {
                      key: 'gen',
                      label: '生成参数',
                      children: (
                        <>
                          <Row gutter={8}>
                            <Col span={12}>
                              <Form.Item name="num_beams" label="num_beams">
                                <InputNumber min={1} max={32} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                            <Col span={12}>
                              <Form.Item name="num_return_sequences" label="num_return_sequences">
                                <InputNumber min={1} max={32} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                          </Row>
                          <Row gutter={8}>
                            <Col span={12}>
                              <Form.Item name="num_beam_groups" label="num_beam_groups">
                                <InputNumber min={1} max={32} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                            <Col span={12}>
                              <Form.Item name="diversity_penalty" label="diversity_penalty">
                                <InputNumber min={0} max={10} step={0.05} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                          </Row>
                          <Row gutter={8}>
                            <Col span={12}>
                              <Form.Item name="do_sample" label="do_sample" valuePropName="checked">
                                <Switch />
                              </Form.Item>
                            </Col>
                            <Col span={12}>
                              <Form.Item name="temperature" label="temperature">
                                <InputNumber min={0} max={2} step={0.05} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                          </Row>
                          <Row gutter={8}>
                            <Col span={12}>
                              <Form.Item name="top_p" label="top_p">
                                <InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                            <Col span={12}>
                              <Form.Item name="max_new_tokens" label="max_new_tokens">
                                <InputNumber min={1} max={4096} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                          </Row>
                          <Form.Item name="seed" label="seed">
                            <InputNumber min={0} max={2 ** 31 - 1} style={{ width: '100%' }} />
                          </Form.Item>
                        </>
                      ),
                    },
                  ]}
                />
              </Form>
            </Card>

            <Card size="small" title="执行日志" style={{ marginTop: 16 }}>
              <pre style={{ background: '#0b0b0b', color: '#ddd', padding: 8, minHeight: 120, maxHeight: 280, overflow: 'auto' }}>
                {logs.join('\n')}
              </pre>
            </Card>
          </Col>
        </Row>
      </Card>
    </Space>
  );
}


