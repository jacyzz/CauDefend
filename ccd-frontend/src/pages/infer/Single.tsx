import { useMemo, useState } from 'react';
import { Button, Card, Col, Divider, Form, Input, InputNumber, Row, Segmented, Select, Space, Switch, Table, Tag, Typography, message } from 'antd';
import CodePane from '../../components/CodePane';
import { inferGenerate } from '../../api/infer';
import type { InferGenParams, InferModelCfg, InferPromptCfg } from '../../api/infer';

const { Text } = Typography;

const DTYPE_OPTIONS = [
  { label: 'float16', value: 'float16' },
  { label: 'bfloat16', value: 'bfloat16' },
  { label: 'auto', value: 'auto' },
];

export default function InferSingle() {
  const [form] = Form.useForm();
  const [input, setInput] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [candidates, setCandidates] = useState<{ text: string; score?: number }[]>([]);
  const [decoded, setDecoded] = useState<string[] | undefined>();

  const disableRun = useMemo(() => !input.trim(), [input]);

  const onRun = async () => {
    try {
      const vals = await form.validateFields();
      const model: InferModelCfg = {
        model: vals.model,
        dtype: vals.dtype,
        device_map: vals.device_map ?? 'auto',
        trust_remote_code: !!vals.trust_remote_code,
        low_cpu_mem_usage: !!vals.low_cpu_mem_usage,
        use_safetensors: !!vals.use_safetensors,
        base_model: vals.base_model || undefined,
        peft_adapter: vals.peft_adapter || undefined,
        peft_merge: !!vals.peft_merge,
      };
      const prompt: InferPromptCfg = {
        template_yaml: vals.template_yaml || undefined,
        system_prompt_text: vals.system_prompt_text || undefined,
      };
      const gen: InferGenParams = {
        max_new_tokens: vals.max_new_tokens,
        do_sample: !!vals.do_sample,
        temperature: vals.temperature,
        top_p: vals.top_p,
        num_beams: vals.num_beams,
        num_return_sequences: vals.num_return_sequences,
        num_beam_groups: vals.num_beam_groups,
        diversity_penalty: vals.diversity_penalty,
        seed: vals.seed,
      };
      setLoading(true);
      setLogs([]);
      const res = await inferGenerate({
        input_text: input,
        model,
        prompt,
        gen,
        return_decoded: true,
      });
      const cands = res.candidates.map((t, i) => ({ text: t, score: res.scores?.[i] }));
      setCandidates(cands);
      setDecoded(res.decoded);
      setLogs([`耗时: ${res.elapsed_ms} ms`, ...res.log]);
    } catch (e: any) {
      message.error(e.message ?? '执行失败');
    } finally {
      setLoading(false);
    }
  };

  // 计算代码区高度
  const viewportH = typeof window !== 'undefined' ? window.innerHeight : 900;
  const codeAreaH = Math.max(240, Math.floor(viewportH - 300));

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Card title="Inference · 单条推理">
        {/* 顶部操作条 */}
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          <Button type="primary" onClick={onRun} loading={loading} disabled={disableRun}>
            开始推理
          </Button>
        </div>

        <Divider />

        <Row gutter={16} wrap={false}>
          {/* 左侧：输入与结果 */}
          <Col flex="1 1 auto">
            <Row gutter={[0, 16]}>
              <Col span={24}>
                <CodePane title="输入（代码或文本）" value={input} onChange={setInput} height={Math.floor(codeAreaH / 2) - 8} />
              </Col>
              <Col span={24}>
                <Card size="small" title="候选结果">
                  {candidates.length === 0 ? (
                    <Text type="secondary">暂无结果</Text>
                  ) : (
                    candidates.map((c, idx) => (
                      <div key={idx} style={{ marginBottom: 16 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                          <Tag color="blue">#{idx}</Tag>
                          {typeof c.score === 'number' && <Text type="secondary">score={c.score.toFixed(4)}</Text>}
                        </div>
                        <CodePane title={`候选 ${idx}`} value={c.text} readOnly height={200} />
                      </div>
                    ))
                  )}
                </Card>
                {decoded && decoded.length > 0 && (
                  <>
                    <Divider />
                    <Card size="small" title="原始解码（调试）">
                      <pre style={{ background: '#0b0b0b', color: '#ddd', padding: 8, maxHeight: 280, overflow: 'auto' }}>
                        {decoded.join('\n\n-----\n\n')}
                      </pre>
                    </Card>
                  </>
                )}
              </Col>
            </Row>
          </Col>

          {/* 右侧：模型、模板与生成参数 */}
          <Col flex="420px">
            <Card size="small" title="模型设置" style={{ marginBottom: 16 }}>
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
                  num_beam_groups: 1,
                  diversity_penalty: 0.0,
                  max_new_tokens: 512,
                  seed: 123456,
                }}
              >
                <Form.Item name="model" label="模型路径/名称" rules={[{ required: true }]}>
                  <Input placeholder="/path/to/merged-or-base-model" />
                </Form.Item>
                <Form.Item name="base_model" label="底座模型（可选，用于LoRA）">
                  <Input placeholder="/path/to/base-model (PEFT时必填或可自动推断)" />
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
                <Text strong>提示模板</Text>
                <Form.Item name="template_yaml" label="模板 YAML（可选）">
                  <Input placeholder="/path/to/template.yaml（含 {{ system_prompt }} 占位符）" />
                </Form.Item>
                <Form.Item name="system_prompt_text" label="系统提示文本">
                  <Input.TextArea rows={3} placeholder="默认：代码重构安全提示；此处可自定义更像 LLaMA-Factory 风格" />
                </Form.Item>

                <Divider />
                <Text strong>生成参数（Beam Search）</Text>
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

            <Card size="small" title="执行日志">
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


