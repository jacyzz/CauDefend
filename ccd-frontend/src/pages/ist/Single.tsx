import { useEffect, useMemo, useState } from 'react';
import { Button, Card, Col, Divider, Form, Row, Select, Space, Tag, Typography, message } from 'antd';
import CodePane from '../../components/CodePane';
import { fetchStyles, transformText } from '../../api/ist';
import type { StyleItem, TransformTextReq } from '../../api/ist';
import StylePicker from '../../components/StylePicker';

const { Text } = Typography;

const LANGS = ['python', 'java', 'c', 'cpp', 'javascript', 'go', 'php'];

export default function IstSinglePage() {
  const [form] = Form.useForm();
  const [input, setInput] = useState<string>('');
  const [output, setOutput] = useState<string>('');
  const [styles, setStyles] = useState<StyleItem[]>([]);
  const [picked, setPicked] = useState<string[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [manual, setManual] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const language = Form.useWatch('language', form) ?? 'python';

  useEffect(() => {
    (async () => {
      try {
        const s = await fetchStyles();
        setStyles(s);
      } catch (e: any) {
        message.error(e.message ?? '加载风格失败');
      }
    })();
  }, []);

  const reloadStyles = async () => {
    try {
      const s = await fetchStyles();
      setStyles(s);
      message.success(`已加载风格：${s.length} 个`);
    } catch (e: any) {
      message.error(e.message ?? '加载风格失败');
    }
  };

  const disableRun = useMemo(() => {
    if (!input.trim()) return true;
    const effective = Array.from(new Set([...(picked || []), ...(manual || [])])).filter(Boolean);
    return effective.length === 0;
  }, [input, picked, manual]);

  const onRun = async () => {
    try {
      const vals = await form.validateFields();
      const effective = Array.from(new Set([...(picked || []), ...(manual || [])])).filter(Boolean);
      if (effective.length === 0) {
        message.warning('请至少选择一个风格');
        return;
      }
      const body: TransformTextReq = {
        language: vals.language,
        code: input,
        strategy: 'fixed',
        styles: effective,
      };
      setLoading(true);
      setLogs([]);
      const res = await transformText(body);
      setOutput(res.converted_code);
      setLogs([`已应用风格: ${res.applied_styles.join(', ')}`, `语法检查通过: ${res.syntax_ok}`, ...res.log]);
    } catch (e: any) {
      message.error(e.message ?? '执行失败');
    } finally {
      setLoading(false);
    }
  };

  // 计算代码区高度
  const viewportH = typeof window !== 'undefined' ? window.innerHeight : 900;
  const codeAreaH = Math.max(240, Math.floor(viewportH - 300));
  const effectiveStyles = Array.from(new Set([...(picked || []), ...(manual || [])])).filter(Boolean);

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Card title="IST · 单一风格转换">
        {/* 顶部控制条 */}
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          <Form form={form} layout="inline" initialValues={{ language: 'python' }}>
            <Form.Item name="language" label="语言" rules={[{ required: true }]}>
              <Select style={{ width: 180 }} options={LANGS.map((l) => ({ label: l, value: l }))} />
            </Form.Item>
          </Form>
          <div style={{ flex: 1 }} />
          <Button onClick={reloadStyles}>刷新风格</Button>
          <Button type="primary" onClick={onRun} loading={loading} disabled={disableRun}>
            开始转换
          </Button>
        </div>
        <Divider />

        <Row gutter={16} wrap={false}>
          {/* 左侧：上下分区的代码输入/输出 */}
          <Col flex="1 1 auto">
            <Row gutter={[0, 16]}>
              <Col span={24}>
                <CodePane title={`输入代码（${language}）`} value={input} onChange={setInput} height={Math.floor(codeAreaH / 2) - 8} />
              </Col>
              <Col span={24}>
                <CodePane title="输出代码" value={output} readOnly height={Math.floor(codeAreaH / 2) - 8} />
              </Col>
            </Row>
          </Col>

          {/* 右侧：风格选择与日志 */}
          <Col flex="360px">
            <Card size="small" title="风格选择" style={{ marginBottom: 16, maxHeight: codeAreaH, overflow: 'auto' }}>
              <Text strong>从列表选择（多选）</Text>
              <div style={{ marginTop: 8 }}>
                <StylePicker styles={styles} value={picked} onChange={setPicked} />
              </div>
              <Divider />
              <Text strong>手动输入风格代码（可选）</Text>
              <Select
                mode="tags"
                style={{ width: '100%', marginTop: 8 }}
                placeholder="输入风格代码后回车；将与上方选择合并"
                value={manual}
                onChange={(v) => setManual(v as string[])}
              />
              {effectiveStyles.length > 0 && (
                <>
                  <Divider />
                  <Text type="secondary">将应用以下风格：</Text>
                  <div style={{ marginTop: 6 }}>
                    {effectiveStyles.map((s) => (
                      <Tag key={s} color="blue" style={{ marginBottom: 6 }}>
                        {s}
                      </Tag>
                    ))}
                  </div>
                </>
              )}
            </Card>

            <Card size="small" title="执行日志">
              <pre style={{ background: '#0b0b0b', color: '#ddd', padding: 8, minHeight: 120, maxHeight: 280, overflow: 'auto' }}>
                {logs.join('\\n')}
              </pre>
            </Card>
          </Col>
        </Row>
      </Card>
    </Space>
  );
}


