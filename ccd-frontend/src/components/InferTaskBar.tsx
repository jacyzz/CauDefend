import { useEffect, useState } from 'react';
import { Space, Typography } from 'antd';
import { LoadingOutlined, CheckCircleOutlined, CloseCircleOutlined, UnorderedListOutlined } from '@ant-design/icons';
import { listInferTasks, type InferTaskSummary } from '../api/infer';

const { Text } = Typography;

interface InferTaskBarProps {
  onClick: () => void;
}

export default function InferTaskBar({ onClick }: InferTaskBarProps) {
  const [tasks, setTasks] = useState<InferTaskSummary[]>([]);
  
  const refresh = async () => {
    try {
      const res = await listInferTasks();
      setTasks(res.tasks || []);
    } catch {
      // silent fail
    }
  };

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, []);

  const running = tasks.filter(t => ['pending','queued','loading_model','running','cancelling'].includes(t.status)).length;
  const completed = tasks.filter(t => t.status === 'completed').length;
  const error = tasks.filter(t => t.status === 'error').length;
  const total = tasks.length;

  if (total === 0) return null;

  return (
    <div 
      onClick={onClick}
      style={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        height: 36,
        background: '#001529',
        borderTop: '1px solid #333',
        color: '#e6f7ff',
        display: 'flex',
        alignItems: 'center',
        padding: '0 24px',
        justifyContent: 'space-between',
        cursor: 'pointer',
        zIndex: 2000,
        boxShadow: '0 -2px 8px rgba(0,0,0,0.15)'
      }}
    >
      <Space size="large">
        <Space>
           <UnorderedListOutlined />
           <Text style={{ color: '#e6f7ff' }} strong>Tasks</Text>
        </Space>
        
        {running > 0 ? (
          <Space>
             <LoadingOutlined spin />
             <Text style={{ color: '#69c0ff' }}>Running: {running}</Text>
          </Space>
        ) : (
          <Text style={{ color: '#8c8c8c' }}>Idle</Text>
        )}

        {(completed > 0 || error > 0) && (
            <Space style={{ marginLeft: 16 }}>
                {completed > 0 && (
                    <Space size="small">
                        <CheckCircleOutlined style={{ color: '#73d13d' }} />
                        <Text style={{ color: '#b7eb8f' }}>{completed}</Text>
                    </Space>
                )}
                {error > 0 && (
                    <Space size="small">
                        <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
                        <Text style={{ color: '#ffa39e' }}>{error}</Text>
                    </Space>
                )}
            </Space>
        )}
      </Space>
      <div style={{ fontSize: 12, color: '#8c8c8c' }}>
         Click to details
      </div>
    </div>
  );
}

