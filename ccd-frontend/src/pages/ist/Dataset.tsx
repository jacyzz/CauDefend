import { useEffect, useState } from 'react';
import { Button, Card, Col, Divider, Form, Input, Row, Select, Space, Switch, Table, Typography, message, Progress } from 'antd';
import { fetchStyles, transformDataset, transformDatasetAsync, fetchTransformProgress } from '../../api/ist';
import type { StyleItem } from '../../api/ist';
import StylePicker from '../../components/StylePicker';

// use options prop for AntD v5
const { Text } = Typography;
const LANGS = ['python', 'java', 'c', 'cpp', 'javascript', 'go', 'php'];

export default function IstDatasetPage() {
  const [form] = Form.useForm();
  const [allStyles, setAllStyles] = useState<StyleItem[]>([]);
  const [pool, setPool] = useState<string[]>([]); // poison_candidates
  const [selected, setSelected] = useState<string[]>([]); // explicit styles
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<any[]>([]);
  const [log, setLog] = useState<any[]>([]);
  const [summary, setSummary] = useState<{ total: number; changed: number; success: number; output_path: string }>();
  const [taskId, setTaskId] = useState<string | null>(null);
  const [percent, setPercent] = useState<number>(0);
  const [status, setStatus] = useState<'normal' | 'active' | 'success' | 'exception'>('normal');
  const [polling, setPolling] = useState<any>(null);
  const useRandom: boolean = (Form.useWatch('use_random', form) as boolean) ?? false;
  const useCombine: boolean = (Form.useWatch('combine_fields', form) as boolean) ?? false;

  useEffect(() => {
    // hydrate persisted
    try {
      const savedForm = localStorage.getItem('ist_dataset_form');
      if (savedForm) form.setFieldsValue(JSON.parse(savedForm));
      const savedPool = localStorage.getItem('ist_dataset_pool');
      if (savedPool) setPool(JSON.parse(savedPool));
      const savedSelected = localStorage.getItem('ist_dataset_selected');
      if (savedSelected) setSelected(JSON.parse(savedSelected));
    } catch {}
    fetchStyles().then(setAllStyles).catch((e) => message.error(e.message));
    return () => {
      try {
        localStorage.setItem('ist_dataset_form', JSON.stringify(form.getFieldsValue()));
        localStorage.setItem('ist_dataset_pool', JSON.stringify(pool));
        localStorage.setItem('ist_dataset_selected', JSON.stringify(selected));
      } catch {}
    };
  }, []);

  const onValuesChange = (_: any, all: any) => {
    try {
      localStorage.setItem('ist_dataset_form', JSON.stringify(all));
    } catch {}
  };
  useEffect(() => {
    try {
      localStorage.setItem('ist_dataset_pool', JSON.stringify(pool));
    } catch {}
  }, [pool]);
  useEffect(() => {
    try {
      localStorage.setItem('ist_dataset_selected', JSON.stringify(selected));
    } catch {}
  }, [selected]);

  const onRun = async () => {
    try {
      const v = await form.validateFields();
      if (!useRandom && selected.length === 0) {
        message.warning('请选择要应用的固定风格');
        return;
      }
      if (useRandom && pool.length === 0) {
        message.warning('请先选择随机采样的风格池');
        return;
      }
      setLoading(true);
      const body: any = {
        input_path: v.input_path,
        output_path: v.output_path,
        language: v.language,
        code_field: v.code_field,
        combine_fields: !!v.combine_fields,
        prompt_field: v.prompt_field || undefined,
        output_prompt_field: v.output_prompt_field || undefined,
        output_code_field: v.output_code_field || undefined,
        id_field: v.id_field || undefined,
        backup_field: v.backup_field || undefined,
        strategy: useRandom ? 'random' : 'fixed',
        styles: useRandom ? undefined : selected,
        poison_min: useRandom ? v.poison_min : undefined,
        poison_max: useRandom ? v.poison_max : undefined,
        avoid_similar: useRandom ? (v.avoid_similar ?? true) : undefined,
        poison_candidates: useRandom ? pool : undefined,
        limit: Number(v.limit || 0),
        seed: v.seed ? Number(v.seed) : undefined,
        syntax_check: !!v.syntax_check,
      };
      const res = await transformDataset(body);
      setSummary({ total: res.total, changed: res.changed, success: res.success, output_path: res.output_path });
      setPreview(res.preview || []);
      setLog(res.log || []);
      message.success(`完成：total=${res.total}  changed=${res.changed}  success=${res.success}`);
    } catch (e: any) {
      message.error(e.message ?? '执行失败');
    } finally {
      setLoading(false);
    }
  };

  const onRunAsync = async () => {
    try {
      const v = await form.validateFields();
      if (!useRandom && selected.length === 0) {
        message.warning('请选择要应用的固定风格');
        return;
      }
      if (useRandom && pool.length === 0) {
        message.warning('请先选择随机采样的风格池');
        return;
      }
      setLoading(true);
      setPercent(0);
      setStatus('active');
      setSummary(undefined);
      setPreview([]);
      setLog([]);
      const body: any = {
        input_path: v.input_path,
        output_path: v.output_path,
        language: v.language,
        code_field: v.code_field,
        combine_fields: !!v.combine_fields,
        prompt_field: v.prompt_field || undefined,
        output_prompt_field: v.output_prompt_field || undefined,
        output_code_field: v.output_code_field || undefined,
        id_field: v.id_field || undefined,
        backup_field: v.backup_field || undefined,
        strategy: useRandom ? 'random' : 'fixed',
        styles: useRandom ? undefined : selected,
        poison_min: useRandom ? v.poison_min : undefined,
        poison_max: useRandom ? v.poison_max : undefined,
        avoid_similar: useRandom ? (v.avoid_similar ?? true) : undefined,
        poison_candidates: useRandom ? pool : undefined,
        limit: Number(v.limit || 0),
        seed: v.seed ? Number(v.seed) : undefined,
        syntax_check: !!v.syntax_check,
      };
      const start = await transformDatasetAsync(body);
      setTaskId(start.task_id);
      setPercent(0);
      setStatus('active');
      const t = setInterval(async () => {
        try {
          const p = await fetchTransformProgress(start.task_id);
          setPercent(Math.max(0, Math.min(100, p.percent)));
          if (p.status === 'done' && p.result) {
            setStatus('success');
            setSummary({
              total: p.result.total,
              changed: p.result.changed,
              success: p.result.success,
              output_path: p.result.output_path,
            });
            setPreview(p.result.preview || []);
            setLog(p.result.log || []);
            clearInterval(t);
            setPolling(null);
            setTaskId(null);
            setLoading(false);
          } else if (p.status === 'error') {
            setStatus('exception');
            message.error(p.error || '处理失败');
            clearInterval(t);
            setPolling(null);
            setTaskId(null);
            setLoading(false);
          }
        } catch (e: any) {
          setStatus('exception');
          message.error(e.message ?? '进度查询失败');
          clearInterval(t);
          setPolling(null);
          setTaskId(null);
          setLoading(false);
        }
      }, 1000);
      setPolling(t);
    } catch (e: any) {
      message.error(e.message ?? '执行失败');
      setLoading(false);
    }
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Card title="IST · 数据集转换">
        <Form
          form={form}
          layout="vertical"
          onValuesChange={onValuesChange}
          initialValues={{ language: 'python', use_random: false, poison_min: 2, poison_max: 3, avoid_similar: true }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="input_path" label="输入 JSONL 路径" rules={[{ required: true }]}>
                <Input placeholder="/abs/path/to/input.jsonl" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="output_path" label="输出 JSONL 路径" rules={[{ required: true }]}>
                <Input placeholder="/abs/path/to/output.jsonl" />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={6}>
              <Form.Item name="language" label="语言" rules={[{ required: true }]}>
                <Select options={LANGS.map((l) => ({ label: l, value: l }))} />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item name="code_field" label="写回字段" rules={[{ required: true }]} initialValue="canonical_solution">
                <Input />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item name="id_field" label="ID 字段">
                <Input placeholder="可选" />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item name="backup_field" label="备份字段">
                <Input placeholder="可选，如 original_code" />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={24}>
              <Form.Item name="combine_fields" label="合并转换 prompt+代码（Humaneval 投毒）" valuePropName="checked" tooltip="先将 prompt 与 code 合并后统一风格转换，再拆分回各自字段">
                <Switch />
              </Form.Item>
            </Col>
            {useCombine && (
              <>
                <Col span={6}>
                  <Form.Item name="prompt_field" label="prompt 字段名" initialValue="prompt">
                    <Input />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item name="output_prompt_field" label="输出 prompt 字段名" tooltip="留空则写回到 prompt 字段">
                    <Input placeholder="默认：prompt_field" />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item name="output_code_field" label="输出代码字段名" tooltip="留空则写回到 写回字段">
                    <Input placeholder="默认：code_field" />
                  </Form.Item>
                </Col>
              </>
            )}
          </Row>

          <Divider />
          <Row gutter={16}>
            <Col span={24}>
              <Form.Item name="use_random" label="启用随机从风格池抽取" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
            {useRandom ? (
              <>
                <Col span={24}>
                  <Text strong>随机候选池（poison_candidates）</Text>
                  <StylePicker styles={allStyles} value={pool} onChange={setPool} />
                </Col>
                <Col span={6}>
                  <Form.Item name="poison_min" label="最少个数">
                    <Input type="number" />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item name="poison_max" label="最多个数">
                    <Input type="number" />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item name="avoid_similar" label="避免同组重复" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
              </>
            ) : (
              <Col span={24}>
                <Text strong>固定应用的风格（多选）</Text>
                <StylePicker styles={allStyles} value={selected} onChange={setSelected} />
              </Col>
            )}
          </Row>

          <Row gutter={16}>
            <Col span={6}>
              <Form.Item name="limit" label="条数上限">
                <Input placeholder="0 表示全部" type="number" />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item name="seed" label="随机种子">
                <Input placeholder="可选" type="number" />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item name="syntax_check" label="语法检查（解析 AST）" valuePropName="checked" tooltip="启用后，会在转换后做一次树解析检查并记录结果">
                <Switch />
              </Form.Item>
            </Col>
            <Col span={12} style={{ display: 'flex', alignItems: 'end', gap: 8 }}>
              <Button type="primary" onClick={onRun} loading={loading}>
                开始转换
              </Button>
              <Button onClick={onRunAsync} disabled={loading}>
                异步转换（显示进度）
              </Button>
              {taskId && (
                <div style={{ minWidth: 280 }}>
                  <Progress percent={Math.round(percent)} status={status} />
                </div>
              )}
              {summary && (
                <span>
                  total: {summary.total} · changed: {summary.changed} · success: {summary.success}
                </span>
              )}
            </Col>
          </Row>
        </Form>

        <Divider />
        <Row gutter={16}>
          <Col span={12}>
            <Card size="small" title="转换后样本（示例）">
              <Table
                size="small"
                dataSource={preview.map((r, i) => ({ key: i, ...r }))}
                pagination={{ pageSize: 5 }}
                scroll={{ x: true }}
              />
            </Card>
          </Col>
          <Col span={12}>
            <Card size="small" title="执行日志">
              <pre style={{ whiteSpace: 'pre-wrap' }}>{log.map((l) => JSON.stringify(l)).join('\n')}</pre>
            </Card>
          </Col>
        </Row>
      </Card>
    </Space>
  );
}


