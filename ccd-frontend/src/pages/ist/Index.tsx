import { Tabs } from 'antd';
import IstSinglePage from './Single';
import IstDatasetPage from './Dataset';

export default function IstHome() {
  return (
    <Tabs
     defaultActiveKey="single"
     items={[
       { key: 'single', label: '单一风格', children: <IstSinglePage /> },
       { key: 'dataset', label: '数据集转换', children: <IstDatasetPage /> },
     ]}
    />
  );
}


