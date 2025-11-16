import { useState } from 'react';
import { Button, Card, Col, Divider, Form, Input, InputNumber, Radio, Row, Select, Space, Switch, Table, Typography, message } from 'antd';
import { inferDataset } from '../../api/infer';
import type { InferDatasetReq, InferGenParams, InferModelCfg, InferPromptCfg } from '../../api/infer';

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

  const onRun = async () => {
    try {
      const v = await form.validateFields();
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
      const body: InferDatasetReq = {
        input_path: v.input_path,
        output_path: v.output_path,
        field: v.field,
        model,
        prompt,
        gen,
        emit_flat: v.emit_flat,
        write_mode: v.write_mode,
        limit: v.limit ?? 0,
      };
      setLoading(true);
      setLogs([]);
      const res = await inferDataset(body);
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
      <Card title="Inference · 数据集推理">
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          <Button type="primary" onClick={onRun} loading={loading}>
            执行数据集推理
          </Button>
        </div>
        <Divider />

        <Row gutter={16} wrap={false}>
          <Col flex="1 1 auto">
            <Card size="small" title="输出预览（最多5条）">
              {preview.length === 0 ? (
                <Text type="secondary">暂无预览</Text>
              ) : (
                <pre style={{ background: '#0b0b0b', color: '#ddd', padding: 8, maxHeight: 560, overflow: 'auto' }}>
                  {preview.map((r, i) => JSON.stringify(r, null, 2)).join('\n\n-----\n\n')}
                </pre>
              )}
            </Card>
          </Col>

          <Col flex="420px">
            <Card size="small" title="数据集配置" style={{ marginBottom: 16 }}>
              <Form
                form={form}
                layout="vertical"
                initialValues={{
                  dtype: 'bfloat16',
                  device_map: 'auto',
                  do_sample: false,
                  temperature: 1.0,
                  top_p: 1.0,
                  num_beams: 4,
                  num_return_sequences: 4,
                  num_beam_groups: 4,
                  diversity_penalty: 0.1,
                  max_new_tokens: 512,
                  seed: 123456,
                  write_mode: 'generation',
                  emit_flat: true,
                }}
              >
                <Form.Item name="input_path" label="输入 JSONL 路径" rules={[{ required: true }]}>
                  <Input placeholder="/path/to/input.jsonl" />
                </Form.Item>
                <Form.Item name="field" label="处理字段" rules={[{ required: true }]}>
                  <Input placeholder="canonical_solution 或 code 等" />
                </Form.Item>
                <Form.Item name="output_path" label="输出 JSONL 路径" rules={[{ required: true }]}>
                  <Input placeholder="/path/to/output.jsonl" />
                </Form.Item>
                <Form.Item name="limit" label="限制条数（0=全部）">
                  <InputNumber min={0} step={1} style={{ width: '100%' }} />
                </Form.Item>
                <Form.Item name="write_mode" label="写入模式">
                  <Select
                    options={[
                      { label: '写入 generation 字段（用于 pass@k）', value: 'generation' },
                      { label: '覆盖原字段（overwrite）', value: 'overwrite' },
                    ]}
                  />
                </Form.Item>
                <Form.Item name="emit_flat" label="多候选展开多行（emit_flat）" valuePropName="checked">
                  <Switch />
                </Form.Item>
              </Form>
            </Card>

            <Card size="small" title="模型与模板" style={{ marginBottom: 16 }}>
              <Form form={form} layout="vertical">
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
                <Form.Item name="dtype" label="dtype">
                  <Select options={DTYPE_OPTIONS} />
                </Form.Item>
                <Form.Item name="device_map" label="device_map">
                  <Input placeholder="auto / cuda:0 / balanced 等" />
                </Form.Item>
                <Form.Item name="trust_remote_code" label="trust_remote_code" valuePropName="checked">
                  <Switch />
                </Form.Item>
                <Form.Item name="low_cpu_mem_usage" label="low_cpu_mem_usage" valuePropName="checked">
                  <Switch />
                </Form.Item>
                <Form.Item name="use_safetensors" label="use_safetensors" valuePropName="checked">
                  <Switch />
                </Form.Item>
                <Divider />
                <Form.Item name="template_yaml" label="模板 YAML（可选）">
                  <Input placeholder="/path/to/template.yaml（含 {{ system_prompt }} 占位符）" />
                </Form.Item>
                <Form.Item name="system_prompt_text" label="系统提示文本">
                  <Input.TextArea rows={3} placeholder="默认代码重构安全提示，可自定义" />
                </Form.Item>
              </Form>
            </Card>

            <Card size="small" title="生成参数（Beam Search）">
              <Form form={form} layout="vertical">
                <Form.Item name="num_beams" label="num_beams">
                  <InputNumber min={1} max={32} style={{ width: '100%' }} />
                </Form.Item>
                <Form.Item name="num_return_sequences" label="num_return_sequences">
                  <InputNumber min={1} max={32} style={{ width: '100%' }} />
                </Form.Item>
                <Form.Item name="num_beam_groups" label="num_beam_groups">
                  <InputNumber min={1} max={32} style={{ width: '100%' }} />
                </Form.Item>
                <Form.Item name="diversity_penalty" label="diversity_penalty">
                  <InputNumber min={0} max={10} step={0.05} style={{ width: '100%' }} />
                </Form.Item>
                <Form.Item name="do_sample" label="do_sample" valuePropName="checked">
                  <Switch />
                </Form.Item>
                <Form.Item name="temperature" label="temperature">
                  <InputNumber min={0} max={2} step={0.05} style={{ width: '100%' }} />
                </Form.Item>
                <Form.Item name="top_p" label="top_p">
                  <InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} />
                </Form.Item>
                <Form.Item name="max_new_tokens" label="max_new_tokens">
                  <InputNumber min={1} max={4096} style={{ width: '100%' }} />
                </Form.Item>
                <Form.Item name="seed" label="seed">
                  <InputNumber min={0} max={2 ** 31 - 1} style={{ width: '100%' }} />
                </Form.Item>
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


