import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card } from '../components/Common/Card';
import { Button } from '../components/Common/Button';
import { SessionService } from '../api/services';
import type { TrackingSession } from '../types';
import { Play, Square, Video, Cpu, Award, Zap, Clock, ArrowLeft, Download, RefreshCw } from 'lucide-react';

export function SessionDetail() {
  const { id } = useParams<{ id: string }>();
  const sessionId = Number(id);
  const navigate = useNavigate();
  
  const [session, setSession] = useState<TrackingSession | null>(null);
  const [liveFrame, setLiveFrame] = useState<string | null>(null);
  const [liveStats, setLiveStats] = useState({ frame_id: 0, active_globals: 0, fps: 0.0 });
  const [outputVideoUrl, setOutputVideoUrl] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [wsStatus, setWsStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const userRole = localStorage.getItem('user_role');
  const canEdit = userRole === 'ADMIN' || userRole === 'OPERATOR';

  const loadSession = async () => {
    try {
      const data = await SessionService.getById(sessionId);
      setSession(data);
      
      if (data.status === 'completed' && !outputVideoUrl) {
        try {
          const outRes = await SessionService.getOutputUrl(sessionId);
          setOutputVideoUrl(outRes.url);
        } catch (e) {
          console.warn('Could not fetch output video URL', e);
        }
      }
    } catch (err) {
      console.error(err);
      setError('Không thể lấy thông tin phiên theo dõi này');
    }
  };

  useEffect(() => {
    loadSession();
    // Poll session status changes if it's created or failed
    const interval = setInterval(() => {
      if (session?.status !== 'running') {
        loadSession();
      }
    }, 4000);
    return () => clearInterval(interval);
  }, [sessionId, session?.status]);

  // WebSocket Connection Handler for live streaming
  useEffect(() => {
    if (session?.status === 'running') {
      connectWebSocket();
    } else {
      disconnectWebSocket();
    }

    return () => {
      disconnectWebSocket();
    };
  }, [session?.status, sessionId]);

  const connectWebSocket = () => {
    if (wsRef.current) return;
    
    setWsStatus('connecting');
    // Using relative WebSocket URL or mapping base URL
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Match the API endpoint config: /api/v1/ws/session/{id}
    const wsUrl = `${wsProto}//localhost:8000/api/v1/ws/session/${sessionId}`;
    
    console.log(`Connecting to stream: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsStatus('connected');
      setError(null);
      console.log('WebSocket stream connected');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'frame') {
          setLiveFrame(`data:image/jpeg;base64,${msg.data.base64_jpg}`);
          setLiveStats({
            frame_id: msg.data.frame_id,
            active_globals: msg.data.active_globals,
            fps: msg.data.fps
          });
        } else if (msg.type === 'session_status') {
          // Status completed/failed triggered by background thread termination
          loadSession();
        }
      } catch (err) {
        console.error('Error reading live stream message', err);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket stream error', err);
    };

    ws.onclose = () => {
      setWsStatus('disconnected');
      wsRef.current = null;
      console.log('WebSocket stream closed');
    };
  };

  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setWsStatus('disconnected');
    }
  };

  const handleStart = async () => {
    if (!canEdit) return;
    setActionLoading(true);
    setError(null);
    try {
      await SessionService.start(sessionId);
      await loadSession();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Bắt đầu tiến trình bám vết thất bại');
    } finally {
      setActionLoading(false);
    }
  };

  const handleStop = async () => {
    if (!canEdit) return;
    if (!window.confirm('Bạn có muốn dừng tiến trình bám vết này?')) return;
    setActionLoading(true);
    setError(null);
    try {
      await SessionService.stop(sessionId);
      await loadSession();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Dừng tiến trình bám vết thất bại');
    } finally {
      setActionLoading(false);
    }
  };

  if (!session) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '300px' }}>
        <RefreshCw className="animate-spin" size={32} color="var(--accent-primary)" />
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-lg)' }}>
      {/* Header breadcrumb */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button 
            onClick={() => navigate('/sessions')}
            style={{
              padding: '8px',
              borderRadius: '8px',
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              color: 'white',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            <ArrowLeft size={18} />
          </button>
          <div>
            <h2 style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.5px', marginBottom: '2px' }}>
              {session.name}
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              Phiên ID: #{session.id} • Khởi tạo: {session.created_at ? new Date(session.created_at).toLocaleDateString('vi-VN') : 'N/A'}
            </p>
          </div>
        </div>

        {canEdit && (
          <div style={{ display: 'flex', gap: '10px' }}>
            {session.status !== 'running' && session.status !== 'stopping' ? (
              <Button 
                onClick={handleStart} 
                disabled={actionLoading}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  backgroundColor: 'var(--success)',
                  border: 'none',
                  boxShadow: '0 4px 12px rgba(34, 197, 94, 0.3)'
                }}
              >
                <Play size={16} fill="white" />
                Bắt đầu xử lý bám vết
              </Button>
            ) : (
              <Button 
                onClick={handleStop} 
                disabled={actionLoading || session.status === 'stopping'}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  backgroundColor: 'var(--error)',
                  border: 'none',
                  boxShadow: '0 4px 12px rgba(239, 68, 68, 0.3)'
                }}
              >
                <Square size={16} fill="white" />
                Dừng phiên làm việc
              </Button>
            )}
          </div>
        )}
      </div>

      {error && (
        <div style={{ padding: '0.75rem 1rem', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', color: 'var(--error)', fontSize: '0.9rem' }}>
          {error}
        </div>
      )}

      {/* Main Layout Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 'var(--spacing-lg)' }}>
        {/* Left pane: Live view or Player */}
        <Card style={{ padding: '0', overflow: 'hidden', display: 'flex', flexDirection: 'column', backgroundColor: '#0c0e12', border: '1px solid var(--border-color)', borderRadius: '12px' }}>
          {session.status === 'running' ? (
            <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' }}>
              {liveFrame ? (
                <img 
                  src={liveFrame} 
                  alt="Live tracking grid view" 
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                />
              ) : (
                <div style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                  <RefreshCw className="animate-spin" size={36} style={{ marginBottom: '12px', animationDuration: '2s' }} />
                  <p style={{ fontSize: '0.95rem' }}>Đang kết nối camera workers và tạo luồng video...</p>
                </div>
              )}
              {/* Live Tag */}
              <div style={{
                position: 'absolute',
                top: '16px',
                left: '16px',
                backgroundColor: 'rgba(239, 68, 68, 0.85)',
                color: 'white',
                fontSize: '0.75rem',
                fontWeight: 700,
                padding: '4px 10px',
                borderRadius: '4px',
                textTransform: 'uppercase',
                letterSpacing: '1px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'white', display: 'inline-block' }} className="animate-ping" />
                LIVE STREAMING
              </div>
            </div>
          ) : session.status === 'completed' ? (
            <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9', backgroundColor: '#000' }}>
              {outputVideoUrl ? (
                <video 
                  src={outputVideoUrl}
                  controls
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                />
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-secondary)' }}>
                  <Video size={48} style={{ marginBottom: '12px', opacity: 0.5 }} />
                  <p>Lưu trữ kết quả đã sẵn sàng. Đang tạo link download...</p>
                </div>
              )}
            </div>
          ) : (
            <div style={{ width: '100%', aspectRatio: '16/9', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', padding: '2rem' }}>
              <Video size={56} style={{ marginBottom: '12px', opacity: 0.4 }} />
              <h4 style={{ color: 'white', fontWeight: 600, fontSize: '1.1rem', marginBottom: '4px' }}>
                {session.status === 'created' ? 'Tiến trình chưa bắt đầu' : session.status === 'stopping' ? 'Đang đóng tiến trình...' : 'Tiến trình bám vết thất bại'}
              </h4>
              <p style={{ fontSize: '0.875rem', textAlign: 'center', maxWidth: '380px' }}>
                {session.status === 'created' 
                  ? 'Vui lòng nhấn nút "Bắt đầu xử lý bám vết" ở phía trên để kích hoạt luồng camera YOLO tracking.' 
                  : session.status === 'stopping' 
                    ? 'Hệ thống đang lưu trữ cơ sở dữ liệu bám vết và giải phóng tài nguyên luồng...' 
                    : 'Đã xảy ra lỗi khi bám vết các nguồn camera. Vui lòng kiểm tra lại cấu hình JSON hiệu chuẩn hoặc video.'}
              </p>
            </div>
          )}
        </Card>

        {/* Right pane: Stats and details */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-lg)' }}>
          {/* Status Box */}
          <Card style={{ padding: 'var(--spacing-md)' }}>
            <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.5px', marginBottom: '12px' }}>
              Trạng thái & Tiến độ
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{
                  width: '12px',
                  height: '12px',
                  borderRadius: '50%',
                  backgroundColor: session.status === 'running' ? 'var(--accent-primary)' : session.status === 'completed' ? 'var(--success)' : session.status === 'failed' ? 'var(--error)' : 'var(--text-secondary)'
                }} />
                <span style={{ fontSize: '1.05rem', fontWeight: 600, textTransform: 'capitalize' }}>
                  {session.status === 'running' ? 'Đang bám vết...' : session.status === 'completed' ? 'Đã hoàn thành' : session.status === 'failed' ? 'Thất bại' : session.status === 'stopping' ? 'Đang dừng...' : 'Chưa kích hoạt'}
                </span>
              </div>

              {session.status === 'running' && (
                <div style={{
                  padding: '8px 12px',
                  backgroundColor: 'rgba(99, 102, 241, 0.08)',
                  border: '1px solid rgba(99, 102, 241, 0.2)',
                  borderRadius: '6px',
                  fontSize: '0.85rem',
                  color: '#818cf8',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  <Cpu size={14} />
                  <span>Luồng WS: {wsStatus === 'connected' ? 'Đang stream mượt mà' : 'Đang thiết lập luồng...'}</span>
                </div>
              )}
            </div>
          </Card>

          {/* Stats Box */}
          <Card style={{ padding: 'var(--spacing-md)' }}>
            <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.5px', marginBottom: '16px' }}>
              Chỉ số bám vết thời gian thực
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  <Award size={16} />
                  Tổng số ID bám vết (ReID)
                </span>
                <span style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent-secondary)' }}>
                  {session.status === 'running' ? liveStats.active_globals : session.total_global_ids}
                </span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  <Clock size={16} />
                  Số Frame đã xử lý
                </span>
                <span style={{ fontSize: '1.1rem', fontWeight: 600 }}>
                  {session.status === 'running' ? liveStats.frame_id : session.total_frames}
                </span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  <Zap size={16} />
                  Tốc độ xử lý (FPS)
                </span>
                <span style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--success)' }}>
                  {session.status === 'running' ? liveStats.fps.toFixed(1) : session.avg_fps.toFixed(1)} FPS
                </span>
              </div>
            </div>
          </Card>

          {/* Download Box */}
          {session.status === 'completed' && outputVideoUrl && (
            <Card style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.5px' }}>
                Tải xuống kết quả
              </h3>
              <a 
                href={outputVideoUrl} 
                download={`session_${sessionId}_output.mp4`}
                target="_blank" 
                rel="noreferrer"
                style={{ width: '100%' }}
              >
                <Button style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  backgroundColor: 'var(--accent-primary)',
                  border: 'none'
                }}>
                  <Download size={18} />
                  Tải Video Annotated kết quả (.mp4)
                </Button>
              </a>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
