import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '../components/Common/Card';
import { Button } from '../components/Common/Button';
import { CameraNetworkService } from '../api/services';
import type { CameraNetwork } from '../types';
import { Eye, Trash2, Plus, Calendar, Film, RefreshCw, X, Settings } from 'lucide-react';

export function CameraNetworks() {
  const [networks, setNetworks] = useState<CameraNetwork[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const navigate = useNavigate();
  const userRole = localStorage.getItem('user_role');
  const canEdit = userRole === 'ADMIN' || userRole === 'OPERATOR';

  const loadData = async () => {
    try {
      const nets = await CameraNetworkService.getAll();
      setNetworks(nets);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000); // Polling for live status updates every 5s
    return () => clearInterval(interval);
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError('Vui lòng điền tên phiên.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const payload = {
        name: name.trim(),
      };
      
      const newSession = await CameraNetworkService.create(payload);
      setShowModal(false);
      setName('');
      loadData();
      navigate(`/camera_networks/${newSession.id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Khởi tạo phiên theo dõi thất bại');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: number) => {
    e.stopPropagation();
    if (!window.confirm('Bạn có chắc muốn xóa phiên theo dõi này? Tất cả dữ liệu lưu trữ liên quan sẽ bị xóa.')) return;
    try {
      await CameraNetworkService.delete(id);
      loadData();
    } catch (err) {
      console.error(err);
    }
  };

  const getStatusStyle = (status: string) => {
    switch (status) {
      case 'running':
        return { backgroundColor: 'rgba(99, 102, 241, 0.15)', border: '1px solid var(--accent-primary)', color: '#818cf8', fontWeight: 600 };
      case 'completed':
        return { backgroundColor: 'rgba(34, 197, 94, 0.15)', border: '1px solid var(--success)', color: 'var(--success)', fontWeight: 600 };
      case 'failed':
        return { backgroundColor: 'rgba(239, 68, 68, 0.15)', border: '1px solid var(--error)', color: 'var(--error)', fontWeight: 600 };
      case 'stopping':
        return { backgroundColor: 'rgba(245, 158, 11, 0.15)', border: '1px solid #f59e0b', color: '#fbbf24', fontWeight: 600 };
      default:
        return { backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', color: 'var(--text-secondary)' };
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'created': return 'Đã tạo';
      case 'running': return 'Đang xử lý';
      case 'stopping': return 'Đang dừng';
      case 'completed': return 'Hoàn thành';
      case 'failed': return 'Thất bại';
      default: return status;
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-lg)' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.5px' }}>Camera Networks (Sessions)</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Quản lý và kích hoạt tiến trình xử lý bám vết theo đối tượng đa camera</p>
        </div>
        
        <div style={{ display: 'flex', gap: '10px' }}>
          <Button onClick={loadData} style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', color: 'white', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <RefreshCw size={16} />
            Làm mới
          </Button>
          {canEdit && (
            <Button onClick={() => setShowModal(true)} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Plus size={18} />
              Tạo phiên theo dõi
            </Button>
          )}
        </div>
      </div>

      {/* Sessions list */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '16px' }}>
        {networks.length === 0 ? (
          <Card style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '4rem', color: 'var(--text-secondary)' }}>
            <Film size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
            <p style={{ fontSize: '1rem' }}>Chưa có phiên theo dõi nào được khởi tạo</p>
            {canEdit && <Button onClick={() => setShowModal(true)} style={{ marginTop: '1rem' }}>Tạo ngay</Button>}
          </Card>
        ) : (
          networks.map((session) => (
            <Card 
              key={session.id}
              onClick={() => navigate(`/camera_networks/${session.id}`)}
              style={{
                padding: 'var(--spacing-lg)',
                cursor: 'pointer',
                display: 'grid',
                gridTemplateColumns: '1.5fr 1fr 1fr 1fr 120px',
                alignItems: 'center',
                gap: '20px',
                transition: 'transform 0.2s ease, border-color 0.2s ease'
              }}
            >
              <div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '6px' }}>{session.name}</h3>
                <div style={{ display: 'flex', gap: '12px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Calendar size={14} />
                    {session.created_at ? new Date(session.created_at).toLocaleString('vi-VN') : 'N/A'}
                  </span>
                  <span>• ID: #{session.id}</span>
                </div>
              </div>

              <div>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Số lượng Camera</span>
                <span style={{ fontSize: '0.95rem', fontWeight: 600 }}>{session.cameras?.length || 0} Cameras</span>
              </div>

              <div>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Độ dài / Thống kê</span>
                <span style={{ fontSize: '0.95rem', fontWeight: 600 }}>
                  {session.total_frames > 0 ? `${session.total_frames} frames • ${session.total_global_ids} IDs` : 'Chưa có thông tin'}
                </span>
              </div>

              <div>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Trạng thái</span>
                <span style={{
                  padding: '4px 10px',
                  borderRadius: '12px',
                  fontSize: '0.8rem',
                  ...getStatusStyle(session.status)
                }}>
                  {getStatusText(session.status)}
                </span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                <Button 
                  onClick={(e) => { e.stopPropagation(); navigate(`/camera_networks/${session.id}?edit=true`); }}
                  style={{
                    padding: '8px',
                    borderRadius: '8px',
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    border: '1px solid var(--accent-primary)',
                    color: '#ffffff'
                  }}
                  title="Cấu hình cameras"
                >
                  <Settings size={18} style={{ strokeWidth: 2.5 }} />
                </Button>
                <Button 
                  onClick={(e) => { e.stopPropagation(); navigate(`/camera_networks/${session.id}`); }}
                  style={{
                    padding: '8px',
                    borderRadius: '8px',
                    backgroundColor: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-color)',
                    color: 'white'
                  }}
                  title="Xem chi tiết"
                >
                  <Eye size={18} />
                </Button>
                {userRole === 'ADMIN' && (
                  <Button 
                    onClick={(e) => handleDelete(e, session.id!)}
                    style={{
                      padding: '8px',
                      borderRadius: '8px',
                      backgroundColor: 'rgba(239, 68, 68, 0.1)',
                      border: '1px solid var(--error)',
                      color: 'var(--error)'
                    }}
                    title="Xóa phiên"
                  >
                    <Trash2 size={18} />
                  </Button>
                )}
              </div>
            </Card>
          ))
        )}
      </div>

      {/* Creation Modal */}
      {showModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1100
        }}>
          <Card style={{
            width: '500px',
            maxHeight: '90vh',
            overflowY: 'auto',
            padding: '2rem',
            position: 'relative',
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)'
          }}>
            <button 
              onClick={() => setShowModal(false)}
              style={{
                position: 'absolute',
                top: '16px',
                right: '16px',
                background: 'transparent',
                border: 'none',
                color: 'var(--text-secondary)',
                cursor: 'pointer'
              }}
            >
              <X size={20} />
            </button>

            <h3 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '1.5rem' }}>Khởi tạo phiên theo dõi mới</h3>

            {error && (
              <div style={{ padding: '0.75rem 1rem', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', color: 'var(--error)', fontSize: '0.85rem', marginBottom: '1rem' }}>
                {error}
              </div>
            )}

            <form onSubmit={handleCreate} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Tên phiên theo dõi</label>
                <input
                  type="text"
                  placeholder="Ví dụ: Phiên Giám sát Sảnh A Sáng 19/5"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    color: 'white',
                    fontSize: '0.9rem'
                  }}
                  required
                />
              </div>

              <div style={{ display: 'flex', gap: '12px', marginTop: '1rem' }}>
                <Button 
                  type="submit" 
                  disabled={loading}
                  style={{
                    flex: 1,
                    height: '44px',
                    background: 'linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))',
                    border: 'none'
                  }}
                >
                  {loading ? 'Đang khởi tạo...' : 'Tạo phiên theo dõi'}
                </Button>
                <Button 
                  type="button"
                  onClick={() => setShowModal(false)}
                  style={{
                    flex: 1,
                    height: '44px',
                    backgroundColor: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-color)',
                    color: 'white'
                  }}
                >
                  Hủy
                </Button>
              </div>
            </form>
          </Card>
        </div>
      )}
    </div>
  );
}
