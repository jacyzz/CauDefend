import { Layout, Menu, theme } from 'antd';
import { CodeOutlined } from '@ant-design/icons';
import { Link, Outlet, useLocation } from 'react-router-dom';

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

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider breakpoint="lg" collapsedWidth="0">
        <div style={{ color: '#fff', padding: '12px 16px', fontWeight: 700 }}>CCD</div>
        <Menu theme="dark" mode="inline" selectedKeys={[selected]} items={items} />
      </Sider>
      <Layout>
        <Header style={{ background: colorBgContainer }} />
        <Content style={{ margin: '16px' }}>
          <div style={{ padding: 16, minHeight: 'calc(100vh - 96px)', background: colorBgContainer }}>
            <Outlet />
          </div>
        </Content>
      </Layout>
    </Layout>
  );
}


