import { Button, Space } from 'antd';
import TextArea from 'antd/es/input/TextArea';
import { CopyOutlined } from '@ant-design/icons';
import { useState } from 'react';

type Props = {
  title?: string;
  value: string;
  onChange?: (v: string) => void;
  readOnly?: boolean;
  rows?: number;
  height?: number; // 优先于 rows
};

export default function CodePane({ title, value, onChange, readOnly, rows = 18, height }: Props) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value ?? '');
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* ignore */
    }
  };
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Space style={{ marginBottom: 8 }}>
        <strong>{title}</strong>
        <Button icon={<CopyOutlined />} size="small" onClick={handleCopy}>
          {copied ? 'Copied' : 'Copy'}
        </Button>
      </Space>
      <TextArea
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        rows={height ? undefined : rows}
        style={{
          fontFamily:
            'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace',
          height: height ? `${height}px` : undefined,
        }}
        readOnly={readOnly}
      />
    </div>
  );
}


