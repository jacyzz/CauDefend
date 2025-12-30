import { useEffect, useMemo, useState } from 'react';
import { Button, Card, Col, Divider, Form, Input, InputNumber, Row, Select, Space, Switch, Tag, Tabs, Typography, message } from 'antd';
import CodePane from '../../components/CodePane';
import { inferGenerate, unloadModel } from '../../api/infer';
import type { InferGenParams, InferModelCfg, InferPromptCfg } from '../../api/infer';

const { Text } = Typography;

const DTYPE_OPTIONS = [
  { label: 'float16', value: 'float16' },
  { label: 'bfloat16', value: 'bfloat16' },
  { label: 'auto', value: 'auto' },
];

export default function InferSingle() {
  const [form] = Form.useForm();
  const provider = Form.useWatch('provider', form) ?? 'local';
  const [input, setInput] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [candidates, setCandidates] = useState<{ text: string; score?: number }[]>([]);
  const [decoded, setDecoded] = useState<string[] | undefined>();
  const [decodeEscapes, setDecodeEscapes] = useState<boolean>(true);
  const [elapsed, setElapsed] = useState<number>(0);
  const [extractSections, setExtractSections] = useState<boolean>(true);

  const fillPreset = () => {
    const vals = form.getFieldsValue();
    form.setFieldsValue({
      num_beams: Math.max(4, vals?.num_beams || 4),
      do_sample: false,
      max_new_tokens: vals?.max_new_tokens || 256,
    });
    message.success('已填充推荐参数（num_beams≥4）。');
  };

  const disableRun = useMemo(() => !input.trim(), [input]);

  // Persist form and input across route/tab switches
  useEffect(() => {
    try {
      const saved = localStorage.getItem('infer_single_form');
      if (saved) {
        const vals = JSON.parse(saved);
        form.setFieldsValue(vals);
      }
      const savedInput = localStorage.getItem('infer_single_input') || '';
      if (savedInput) setInput(savedInput);
    } catch {}
    // unload on leave if enabled
    return () => {
      try {
        const vals = form.getFieldsValue();
        const p = (vals?.provider || 'local') as string;
        if (vals?.unload_on_leave) {
          const model: InferModelCfg | undefined = vals?.model || vals?.model === '' ? undefined : {
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
          if (p === 'local' && model && typeof model.model === 'string' && model.model.trim()) {
            // fire and forget
            unloadModel({ model }).catch(() => void 0);
          }
        }
      } catch {}
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleValuesChange = (_: any, allValues: any) => {
    try {
      localStorage.setItem('infer_single_form', JSON.stringify(allValues));
    } catch {}
  };

  useEffect(() => {
    try {
      localStorage.setItem('infer_single_input', input);
    } catch {}
  }, [input]);

  const onRun = async () => {
    try {
      const vals = await form.validateFields();
      setLoading(true);
      setElapsed(0);
      setLogs([]);
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
      const res = await inferGenerate({
        provider: vals.provider || 'local',
        input_text: input,
        model,
        prompt,
        gen,
        return_decoded: true,
        unload_after: !!vals.unload_after,
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

  // simple elapsed timer while loading
  useEffect(() => {
    let timer: any;
    if (loading) {
      const start = Date.now();
      timer = setInterval(() => {
        setElapsed(Math.max(0, Math.floor((Date.now() - start) / 1000)));
      }, 500);
    } else {
      setElapsed(0);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [loading]);

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Card title="Inference · 单条推理（简化）">
        {/* 顶部操作条 */}
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          <Button type="primary" onClick={onRun} loading={loading} disabled={disableRun}>
            开始推理
          </Button>
          <Button onClick={fillPreset} disabled={loading}>填充推荐参数</Button>
          <Space size="small" style={{ marginLeft: 'auto' }}>
            <Text type="secondary">转义解码显示</Text>
            <Switch checked={decodeEscapes} onChange={setDecodeEscapes} />
            <Text type="secondary" style={{ marginLeft: 8 }}>解析思维链</Text>
            <Switch checked={extractSections} onChange={setExtractSections} />
            {loading && <Text type="secondary">已运行 {elapsed}s</Text>}
          </Space>
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
                    <>
                      {candidates.map((c, idx) => {
                        const decode = (s: string) =>
                          decodeEscapes && typeof s === 'string'
                            ? s.replace(/\\r\\n|\\n/g, '\n').replace(/\\t/g, '\t')
                            : s;

                        const text = decode(c.text || '');

                        const firstFenced = (s: string): string | null => {
                          const m = s.match(/```[a-zA-Z0-9_+\-]*\n([\s\S]*?)```/m);
                          return m ? m[1].trim() : null;
                        };

                        const extract = (s: string): { analysis: string; code: string } => {
                          if (!extractSections) return { analysis: '', code: s };
                          const t = s.replace(/\r\n/g, '\n');
                          // [Trace Analysis] ... [Sanitized Code] ...  (JS RegExp doesn't support (?is), use flags + [\\s\\S])
                          const m1 = t.match(/\[trace\s*analysis\]([\s\S]*)\[sanitized\s*code\]([\s\S]*)$/i);
                          if (m1) {
                            const analysis = m1[1].trim();
                            const tail = m1[2].trim();
                            const code = firstFenced(tail) || tail;
                            return { analysis, code };
                          }
                          // Trace Analysis: ... (Sanitized Code: ...)?
                          const m2 = t.match(/trace\s*analysis\s*:\s*([\s\S]*)$/i);
                          if (m2) {
                            const after = m2[1];
                            const m2b = after.match(/([\s\S]*)sanitized\s*code\s*:\s*([\s\S]*)$/i);
                            if (m2b) {
                              const analysis = m2b[1].trim();
                              const tail = m2b[2].trim();
                              const code = firstFenced(tail) || tail;
                              return { analysis, code };
                            }
                            const block = firstFenced(after);
                            if (block) {
                              const head = after.split('```', 1)[0].trim();
                              return { analysis: head, code: block };
                            }
                          }
                          const only = firstFenced(t);
                          if (only) {
                            const head = t.split('```', 1)[0].trim();
                            return { analysis: head, code: only };
                          }
                          return { analysis: '', code: t };
                        };

                        const { analysis, code } = extract(text);
                        const analysisShown = decode(analysis);
                        const codeShown = decode(code);

                        return (
                          <div key={idx} style={{ border: '1px solid #333', borderRadius: 4, marginBottom: 12 }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: '#111', padding: '4px 6px' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                <Tag color="blue">Variant {idx + 1}</Tag>
                                {typeof c.score === 'number' && <Text type="secondary">score={c.score.toFixed(4)}</Text>}
                              </div>
                            </div>
                            {analysisShown && (
                              <Card size="small" bodyStyle={{ padding: 8 }} style={{ border: 'none' }} title="Trace Analysis">
                                <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{analysisShown}</pre>
                              </Card>
                            )}
                            <CodePane title="Sanitized Code" value={codeShown} readOnly height={260} />
                          </div>
                        );
                      })}
                    </>
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

          {/* 右侧：参数（使用 Tabs 分组，减少滚动） */}
          <Col flex="420px">
            <Card size="small" title="参数设置" style={{ marginBottom: 16 }}>
              <Form
                form={form}
                layout="vertical"
                onValuesChange={handleValuesChange}
                initialValues={{
                  provider: 'local',
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
                  unload_after: false,
                }}
              >
                <Tabs
                  defaultActiveKey="quick"
                  items={[
                    {
                      key: 'quick',
                      label: '快速参数',
                      children: (
                        <>
                          <Space style={{ marginBottom: 8 }}>
                            <Button onClick={fillPreset}>填充推荐参数</Button>
                          </Space>
                          <Form.Item name="provider" label="推理后端 (provider)">
                            <Select
                              options={[
                                { label: '本地 HF (local)', value: 'local' },
                                { label: '本地 vLLM (vllm)', value: 'vllm' },
                                { label: '本地 vLLM#0 (vllm0)', value: 'vllm0' },
                                { label: '本地 vLLM#1 (vllm1)', value: 'vllm1' },
                                { label: '远端 OpenAI兼容 (openai_compatible)', value: 'openai_compatible' },
                                { label: '远端 OpenAI (openai)', value: 'openai' },
                                { label: '远端 DeepSeek (deepseek)', value: 'deepseek' },
                              ]}
                            />
                          </Form.Item>
                          <Form.Item name="model" label="模型路径/名称" rules={[{ required: true }]}>
                            <Input placeholder={provider === 'local' ? '/path/to/merged-or-base-model' : 'remote model name (e.g. gpt-4o-mini / deepseek-chat)'} />
                          </Form.Item>
                          <Row gutter={8}>
                            <Col span={12}>
                              <Form.Item name="dtype" label="dtype">
                                <Select options={DTYPE_OPTIONS} disabled={provider !== 'local'} />
                              </Form.Item>
                            </Col>
                            <Col span={12}>
                              <Form.Item name="device_map" label="device_map">
                                <Input placeholder="auto / cuda:0 / balanced 等" disabled={provider !== 'local'} />
                              </Form.Item>
                            </Col>
                          </Row>
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
                              <Form.Item name="max_new_tokens" label="max_new_tokens">
                                <InputNumber min={1} max={4096} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                            <Col span={12}>
                              <Form.Item name="do_sample" label="do_sample" valuePropName="checked">
                                <Switch />
                              </Form.Item>
                            </Col>
                          </Row>
                          <Row gutter={8}>
                            <Col span={12}>
                              <Form.Item name="temperature" label="temperature">
                                <InputNumber min={0} max={2} step={0.05} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                            <Col span={12}>
                              <Form.Item name="top_p" label="top_p">
                                <InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                          </Row>
                        </>
                      ),
                    },
                    {
                      key: 'model',
                      label: '模型 / LoRA',
                      children: (
                        <>
                          <Form.Item name="base_model" label="底座模型（可选，用于LoRA）">
                            <Input placeholder="/path/to/base-model (PEFT时必填或可自动推断)" disabled={provider !== 'local'} />
                          </Form.Item>
                          <Form.Item name="peft_adapter" label="PEFT 适配器（可选）">
                            <Input placeholder="/path/to/adapter-dir (含 adapter_config.json)" disabled={provider !== 'local'} />
                          </Form.Item>
                          <Form.Item name="peft_merge" label="合并LoRA到权重" valuePropName="checked">
                            <Switch disabled={provider !== 'local'} />
                          </Form.Item>
                          <Row gutter={8}>
                            <Col span={8}>
                              <Form.Item name="trust_remote_code" label="trust_remote_code" valuePropName="checked">
                                <Switch disabled={provider !== 'local'} />
                              </Form.Item>
                            </Col>
                            <Col span={8}>
                              <Form.Item name="low_cpu_mem_usage" label="low_cpu_mem_usage" valuePropName="checked">
                                <Switch disabled={provider !== 'local'} />
                              </Form.Item>
                            </Col>
                            <Col span={8}>
                              <Form.Item name="use_safetensors" label="use_safetensors" valuePropName="checked">
                                <Switch disabled={provider !== 'local'} />
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
                            <Input.TextArea rows={3} placeholder="默认：代码重构安全提示；此处可自定义更像 LLaMA-Factory 风格" />
                          </Form.Item>
                        </>
                      ),
                    },
                    {
                      key: 'advanced',
                      label: '高级生成',
                      children: (
                        <>
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
                          <Form.Item name="seed" label="seed">
                            <InputNumber min={0} max={2 ** 31 - 1} style={{ width: '100%' }} />
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
                  ]}
                />
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


