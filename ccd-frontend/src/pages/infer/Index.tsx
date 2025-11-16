import { Tabs } from 'antd';
import InferSingle from './Single';
import InferDataset from './Dataset';

export default function InferHome() {
  return (
    <Tabs
      defaultActiveKey="single"
      items={[
        { key: 'single', label: '单条推理', children: <InferSingle /> },
        { key: 'dataset', label: '数据集推理', children: <InferDataset /> },
      ]}
    />
  );
}


