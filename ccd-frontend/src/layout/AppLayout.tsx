import { Button, Layout, Menu, theme } from 'antd';
import { CodeOutlined } from '@ant-design/icons';
import { useState } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';

import InferTasksDrawer from '../components/InferTasksDrawer';
import InferTaskBar from '../components/InferTaskBar';

const { Header, Sider, Content } = Layout;

const items = [
  { key: '/ist', icon: <CodeOutlined />, label: <Link to="/ist">IST</Link> },
  { key: '/infer', icon: <CodeOutlined />, label: <Link to="/infer">Inference</Link> },
  // 预留：后续可追加 { key: '/train', ... }, { key: '/chat', ... }
];

export default function AppLayout() {
  const {
    token: { colorBgContainer },
  } = theme.useToken();
  const location = useLocation();
  const selected = items.find((it) => location.pathname.startsWith(it.key))?.key ?? '/';
  const [tasksOpen, setTasksOpen] = useState(false);

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider breakpoint="lg" collapsedWidth="0">
        <div style={{ color: '#fff', padding: '12px 16px', fontWeight: 700 }}>CCD</div>
        <Menu theme="dark" mode="inline" selectedKeys={[selected]} items={items} />
      </Sider>
      <Layout>
        <Header style={{ background: colorBgContainer, display: 'flex', justifyContent: 'flex-end', alignItems: 'center' }}>
          <Button onClick={() => setTasksOpen(true)}>任务</Button>
        </Header>
        <Content style={{ margin: '16px' }}>
          <div style={{ padding: 16, minHeight: 'calc(100vh - 96px)', background: colorBgContainer }}>
            <Outlet />
          </div>
          <div style={{ height: 40 }} />
        </Content>
      </Layout>

      <InferTasksDrawer open={tasksOpen} onClose={() => setTasksOpen(false)} />
      <InferTaskBar onClick={() => setTasksOpen(true)} />
    </Layout>
  );
}


