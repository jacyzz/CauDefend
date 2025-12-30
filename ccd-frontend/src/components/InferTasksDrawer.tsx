import { useEffect, useMemo, useState } from 'react';
import { Button, Drawer, List, Progress, Space, Tag, Typography, message } from 'antd';
import { cancelInferTask, listInferTasks, type InferTaskSummary } from '../api/infer';

const { Text } = Typography;

function statusTagColor(status?: string): string {
  switch (status) {
    case 'completed':
      return 'green';
    case 'running':
      return 'blue';
    case 'loading_model':
      return 'gold';
    case 'queued':
    case 'pending':
      return 'default';
    case 'cancelling':
      return 'orange';
    case 'cancelled':
      return 'default';
    case 'error':
      return 'red';
    default:
      return 'default';
  }
}

function fmtTime(ts?: number): string {
  if (!ts) return '-';
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return String(ts);
  }
}

export default function InferTasksDrawer(props: { open: boolean; onClose: () => void }) {
  const [loading, setLoading] = useState(false);
  const [tasks, setTasks] = useState<InferTaskSummary[]>([]);

  const runningCount = useMemo(
    () => tasks.filter((t) => ['pending', 'queued', 'loading_model', 'running', 'cancelling'].includes(t.status)).length,
    [tasks],
  );

  const refresh = async (quiet = false) => {
    try {
      if (!quiet) setLoading(true);
      const res = await listInferTasks();
      setTasks(res.tasks || []);
    } catch (e: any) {
      if (!quiet) message.error(e?.message ?? '拉取任务列表失败');
    } finally {
      if (!quiet) setLoading(false);
    }
  };

  useEffect(() => {
    if (!props.open) return;
    refresh(true).catch(() => void 0);
    const timer = setInterval(() => {
      refresh(true).catch(() => void 0);
    }, 3000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.open]);

  const onCancel = async (taskId: string) => {
    try {
      await cancelInferTask(taskId);
      message.info('已请求取消');
      await refresh(true);
    } catch (e: any) {
      message.error(e?.message ?? '取消失败');
    }
  };

  return (
    <Drawer
      title={
        <Space>
          <Text strong>推理任务</Text>
          <Tag color={runningCount > 0 ? 'blue' : 'default'}>{runningCount > 0 ? `运行中 ${runningCount}` : '无运行中任务'}</Tag>
        </Space>
      }
      open={props.open}
      onClose={props.onClose}
      width={720}
      extra={
        <Space>
          <Button onClick={() => refresh(false)} loading={loading}>
            刷新
          </Button>
        </Space>
      }
    >
      <List
        dataSource={tasks}
        locale={{ emptyText: '暂无任务（只显示本次后端进程内创建过的任务）' }}
        renderItem={(t) => {
          const canCancel = ['pending', 'queued', 'loading_model', 'running'].includes(t.status);
          const percent = Math.max(0, Math.min(100, Number(t.percent ?? 0)));
          return (
            <List.Item
              actions={[
                <Button key="cancel" danger disabled={!canCancel} onClick={() => onCancel(t.task_id)}>
                  取消
                </Button>,
              ]}
            >
              <List.Item.Meta
                title={
                  <Space wrap>
                    <Text code>{t.task_id}</Text>
                    <Tag color={statusTagColor(t.status)}>{t.status}</Tag>
                    {t.provider ? <Tag>{t.provider}</Tag> : null}
                    {t.model ? <Tag>{t.model}</Tag> : null}
                  </Space>
                }
                description={
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
                      <Text type="secondary">创建: {fmtTime(t.created_at)}</Text>
                      <Text type="secondary">更新: {fmtTime(t.updated_at)}</Text>
                    </div>
                    <Progress
                      percent={Math.round(percent * 10) / 10}
                      status={t.status === 'error' ? 'exception' : t.status === 'completed' ? 'success' : 'active'}
                    />
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <Text type="secondary">输入: {t.input_path || '-'}</Text>
                      <Text type="secondary">输出: {t.output_path || t.result?.output_path || '-'}</Text>
                      {t.error ? <Text type="danger">错误: {t.error}</Text> : null}
                    </div>
                  </div>
                }
              />
            </List.Item>
          );
        }}
      />
    </Drawer>
  );
}
