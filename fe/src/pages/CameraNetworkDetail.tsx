import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card } from '../components/Common/Card';
import { Button } from '../components/Common/Button';
import { CameraNetworkService, CameraService } from '../api/services';
import type { CameraNetwork, Camera } from '../types';
import { Play, Square, Video, Cpu, Award, Zap, Clock, ArrowLeft, Download, RefreshCw, Trash2, Plus, Upload, X, Loader2, Settings } from 'lucide-react';

interface RoiConfig {
  id: string;
  name: string;
  polygon: number[][];
}

export function CameraNetworkDetail() {
  const { id } = useParams<{ id: string }>();
  const networkId = Number(id);
  const navigate = useNavigate();
  
  const [network, setNetwork] = useState<CameraNetwork | null>(null);
  const [liveFrame, setLiveFrame] = useState<string | null>(null);
  const [liveStats, setLiveStats] = useState({ frame_id: 0, active_globals: 0, fps: 0.0 });
  const [outputVideoUrl, setOutputVideoUrl] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [wsStatus, setWsStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [error, setError] = useState<string | null>(null);
  
  // Custom camera management states
  const [thumbnails, setThumbnails] = useState<Record<number, string>>({});
  const [showAddCamModal, setShowAddCamModal] = useState(false);
  const [camName, setCamName] = useState('');
  const [camLocation, setCamLocation] = useState('');
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [calibFile, setCalibFile] = useState<File | null>(null);
  const [uploadStep, setUploadStep] = useState<'idle' | 'creating' | 'uploading_video' | 'uploading_calib' | 'finishing' | 'roi_drawing'>('idle');
  const [uploadProgressText, setUploadProgressText] = useState('');

  // ROI drawing states
  const [createdCamId, setCreatedCamId] = useState<number | null>(null);
  const [editingCamId, setEditingCamId] = useState<number | null>(null);
  const [frameUrl, setFrameUrl] = useState<string>('');
  const [rois, setRois] = useState<RoiConfig[]>([]);
  const [currentPoints, setCurrentPoints] = useState<number[][]>([]);
  const [newRoiName, setNewRoiName] = useState('');
  const [isDrawing, setIsDrawing] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const userRole = localStorage.getItem('user_role');
  const canEdit = userRole === 'ADMIN' || userRole === 'OPERATOR';

  const loadThumbnails = async (cams: Camera[]) => {
    const map: Record<number, string> = {};
    await Promise.all(
      cams.map(async (cam) => {
        if (cam.id) {
          try {
            const res = await CameraService.getThumbnail(cam.id);
            map[cam.id] = res.url;
          } catch (e) {
            console.warn(`Could not get thumbnail for camera ${cam.id}`, e);
          }
        }
      })
    );
    setThumbnails((prev) => ({ ...prev, ...map }));
  };

  const loadNetwork = async () => {
    try {
      const data = await CameraNetworkService.getById(networkId);
      setNetwork(data);
      if (data.cameras) {
        loadThumbnails(data.cameras);
      }
      
      if (data.status === 'completed' && !outputVideoUrl) {
        try {
          const outRes = await CameraNetworkService.getOutputUrl(networkId);
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
    loadNetwork();
    // Poll network status changes only if it's stopping
    const interval = setInterval(() => {
      if (network?.status === 'stopping') {
        loadNetwork();
      }
    }, 4000);
    return () => clearInterval(interval);
  }, [networkId, network?.status]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('edit') === 'true' && network && network.status === 'created') {
      setShowAddCamModal(true);
    }
  }, [network]);

  // WebSocket Connection Handler for live streaming
  useEffect(() => {
    if (network?.status === 'running' || network?.status === 'stopping') {
      connectWebSocket();
    } else {
      disconnectWebSocket();
    }

    return () => {
      disconnectWebSocket();
    };
  }, [network?.status, networkId]);

  const connectWebSocket = () => {
    if (wsRef.current) return;
    
    setWsStatus('connecting');
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${window.location.host}/api/v1/ws/network/${networkId}`;
    
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
          loadNetwork();
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
    if (!network?.cameras || network.cameras.length === 0) {
      setError('Vui lòng thêm ít nhất một camera có đầy đủ cấu hình trước khi bắt đầu.');
      return;
    }
    setActionLoading(true);
    setError(null);
    try {
      await CameraNetworkService.start(networkId);
      await loadNetwork();
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
      await CameraNetworkService.stop(networkId);
      await loadNetwork();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Dừng tiến trình bám vết thất bại');
    } finally {
      setActionLoading(false);
    }
  };

  const handleEditCamera = async (cam: Camera) => {
    setEditingCamId(cam.id || null);
    setCamName(cam.name || '');
    setCamLocation(cam.location || '');
    setVideoFile(null);
    setCalibFile(null);
    setCreatedCamId(cam.id || null);
    setRois([]);
    setCurrentPoints([]);
    setShowAddCamModal(true);

    if (cam.id) {
      // Load frame for ROI drawing
      try {
        const { apiClient } = await import('../api/client');
        const response = await apiClient.get(`/cameras/${cam.id}/frame`, { responseType: 'blob' });
        const url = URL.createObjectURL(response.data);
        setFrameUrl(url);
      } catch (err) {
        console.error('Error loading camera frame:', err);
      }

      // Load existing ROIs
      try {
        const { CameraRoiService } = await import('../api/services');
        const currentRois = await CameraRoiService.getRois(cam.id);
        setRois(currentRois);
      } catch (err) {
        console.error('Error loading camera ROIs:', err);
      }

      // Jump directly to combined edit mode (skip the upload-first step)
      setUploadStep('roi_drawing');
      setUploadProgressText('');
    } else {
      setUploadStep('idle');
    }
  };

  const handleSaveCamera = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!camName.trim()) {
      setError('Vui lòng nhập tên camera.');
      return;
    }
    if (!videoFile && !editingCamId) {
      setError('Vui lòng chọn file video nguồn.');
      return;
    }

    setError(null);
    setUploadStep('creating');
    setUploadProgressText(editingCamId ? 'Đang cập nhật thông tin camera...' : 'Đang tạo thực thể camera trên hệ thống...');

    try {
      let activeCamId = editingCamId;
      if (editingCamId) {
        // Update camera details
        await CameraService.update(editingCamId, {
          name: camName.trim(),
          location: camLocation.trim() || null,
          ...(videoFile ? { source: `videos/${videoFile.name}` } : {})
        });
      } else {
        // 1. Create camera in DB associated to this network
        const newCam = await CameraService.create({
          name: camName.trim(),
          network_id: networkId,
          source: `videos/${videoFile!.name}`,
          source_type: 'video',
          location: camLocation.trim() || undefined,
          is_active: false
        });

        if (!newCam.id) {
          throw new Error('Không nhận được ID camera từ máy chủ');
        }
        activeCamId = newCam.id;
        setCreatedCamId(newCam.id);
      }

      if (!activeCamId) {
        throw new Error('ID camera không hợp lệ');
      }

      // 2. Upload video file if provided
      if (videoFile) {
        setUploadStep('uploading_video');
        setUploadProgressText('Đang tải lên file video và tự động trích xuất metadata (có thể mất vài giây)...');
        await CameraService.uploadVideo(activeCamId, videoFile!);
      }

      // 3. Upload calibration matrix if present
      if (calibFile) {
        setUploadStep('uploading_calib');
        setUploadProgressText('Đang tải lên và cấu hình ma trận hiệu chuẩn camera...');
        await CameraService.uploadCalibration(activeCamId, calibFile);
      }

      // 4. Prepare for ROI drawing
      setUploadStep('finishing');
      setUploadProgressText('Đang tải khung hình mẫu để vẽ vùng giám sát (ROI)...');
      
      try {
        const { apiClient } = await import('../api/client');
        const response = await apiClient.get(`/cameras/${activeCamId}/frame`, { responseType: 'blob' });
        const url = URL.createObjectURL(response.data);
        setFrameUrl(url);
      } catch (err) {
        console.error('Error loading camera frame:', err);
      }

      setUploadStep('roi_drawing');
      setUploadProgressText('');

    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || 'Lỗi trong quá trình khởi tạo camera. Vui lòng kiểm tra lại file.');
      setUploadStep('idle');
      setUploadProgressText('');
    }
  };

  const finishAddCamera = async () => {
    // Reset states
    setCamName('');
    setCamLocation('');
    setVideoFile(null);
    setCalibFile(null);
    setCreatedCamId(null);
    setEditingCamId(null);
    setFrameUrl('');
    setRois([]);
    setCurrentPoints([]);
    setUploadStep('idle');
    setShowAddCamModal(false);
    
    // Reload network details
    await loadNetwork();
  };

  const getCameraResolution = () => {
    if (imageRef.current && imageRef.current.naturalWidth && imageRef.current.naturalHeight) {
      return { width: imageRef.current.naturalWidth, height: imageRef.current.naturalHeight };
    }
    // Fallback to active camera's resolution if found
    const activeCam = network?.cameras?.find(c => c.id === (createdCamId || editingCamId));
    if (activeCam && activeCam.resolution) {
      const parts = activeCam.resolution.split('x').map(Number);
      if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
        return { width: parts[0], height: parts[1] };
      }
    }
    return { width: 1920, height: 1080 };
  };

  const startDrawingWorkflow = () => {
    const name = prompt("Nhập tên khu vực ROI (ví dụ: Cửa chính, Khu vực A):");
    if (name === null) return; // Cancelled
    setNewRoiName(name.trim() || `Khu vực ${rois.length + 1}`);
    setIsDrawing(true);
    setCurrentPoints([]);
  };

  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isDrawing) return;
    if (!imageRef.current) return;
    const rect = imageRef.current.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top) / rect.height;
    setCurrentPoints([...currentPoints, [nx, ny]]);
  };

  const handleAddRoi = () => {
    if (currentPoints.length < 3) {
      alert('Vui lòng vẽ ít nhất 3 điểm để tạo một đa giác (Polygon).');
      return;
    }
    const { width, height } = getCameraResolution();
    const pixelPolygon = currentPoints.map(([nx, ny]) => [
      Math.round(nx * width),
      Math.round(ny * height)
    ]);
    const newRoi: RoiConfig = {
      id: `roi_${Date.now()}`,
      name: newRoiName.trim() || `Khu vực ${rois.length + 1}`,
      polygon: pixelPolygon
    };
    setRois([...rois, newRoi]);
    setCurrentPoints([]);
    setNewRoiName('');
    setIsDrawing(false);
  };

  const handleDeleteRoi = (id: string) => {
    setRois(rois.filter(r => r.id !== id));
  };

  const handleSaveRois = async () => {
    const activeCamId = createdCamId || editingCamId;
    if (!activeCamId) {
      await finishAddCamera();
      return;
    }

    try {
      // If editing, also update camera name/location and upload files
      if (editingCamId) {
        // Update camera name and location
        if (camName.trim()) {
          await CameraService.update(editingCamId, {
            name: camName.trim(),
            location: camLocation.trim() || null,
          });
        }

        // Upload video if a new one was selected
        if (videoFile) {
          await CameraService.uploadVideo(editingCamId, videoFile);
        }

        // Upload calibration JSON if a new one was selected
        if (calibFile) {
          await CameraService.uploadCalibration(editingCamId, calibFile);
        }
      }

      // Save ROIs
      const { CameraRoiService } = await import('../api/services');
      await CameraRoiService.saveRois(activeCamId, rois);
    } catch (err) {
      console.error(err);
      alert('Không thể lưu cấu hình. Vui lòng thử lại.');
      return;
    }
    await finishAddCamera();
  };

  const renderPolygonPoints = (pts: number[][]) => {
    const { width, height } = getCameraResolution();
    return pts.map(([nx, ny]) => `${nx * width},${ny * height}`).join(' ');
  };

  const getPixelPoints = (pixelPoly: number[][]) => {
    if (!pixelPoly) return '';
    return pixelPoly.map(([px, py]) => `${px},${py}`).join(' ');
  };

  const handleDeleteCamera = async (camId: number) => {
    if (!window.confirm('Bạn có chắc chắn muốn xóa camera này? Tất cả video và cấu hình liên quan đến camera này sẽ bị xóa khỏi hệ thống.')) {
      return;
    }
    setActionLoading(true);
    setError(null);
    try {
      await CameraService.delete(camId);
      await loadNetwork();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Xóa camera thất bại');
    } finally {
      setActionLoading(false);
    }
  };

  const getGridAspectRatio = () => {
    const sortedCamIds = (network?.cameras || []).map(c => c.id!).sort((a, b) => a - b);
    const numCams = sortedCamIds.length;
    if (numCams === 0) return '16/9';
    const numCols = numCams === 1 ? 1 : 2;
    const numRows = Math.ceil(numCams / numCols);
    
    let camWidth = 1920;
    let camHeight = 1080;
    const firstCam = network?.cameras?.[0];
    if (firstCam && firstCam.resolution) {
      const parts = firstCam.resolution.split('x').map(Number);
      if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
        camWidth = parts[0];
        camHeight = parts[1];
      }
    }
    return `${numCols * camWidth} / ${numRows * camHeight}`;
  };

  if (!network) {
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
            onClick={() => navigate('/camera_networks')}
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
              {network.name}
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              Camera Network ID: #{network.id} • Khởi tạo: {network.created_at ? new Date(network.created_at).toLocaleDateString('vi-VN') : 'N/A'}
            </p>
          </div>
        </div>

        {canEdit && (
          <div style={{ display: 'flex', gap: '10px' }}>
            <Button 
              onClick={loadNetwork} 
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                backgroundColor: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                color: 'white'
              }}
            >
              <RefreshCw size={16} />
              Làm mới
            </Button>
            {network.status !== 'running' && network.status !== 'stopping' ? (
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
                disabled={actionLoading || network.status === 'stopping'}
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
      <div style={{ display: 'grid', gridTemplateColumns: '2.2fr 1fr', gap: 'var(--spacing-lg)' }}>
        {/* Left pane: Live view or Player */}
        <Card style={{ padding: network.status === 'created' ? 'var(--spacing-lg)' : '0', overflow: 'hidden', display: 'flex', flexDirection: 'column', backgroundColor: '#0c0e12', border: '1px solid var(--border-color)', borderRadius: '12px' }}>
          {network.status === 'running' ? (
            <div style={{ position: 'relative', width: '100%', aspectRatio: getGridAspectRatio(), display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' }}>
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
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'white', display: 'inline-block' }} />
                LIVE STREAMING
              </div>
            </div>
          ) : network.status === 'completed' ? (
            <div style={{ position: 'relative', width: '100%', aspectRatio: getGridAspectRatio(), backgroundColor: '#000' }}>
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
          ) : network.status === 'created' ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'white', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Video size={18} color="var(--accent-primary)" />
                  Mạng lưới camera thành viên ({network.cameras?.length || 0})
                </h4>
              </div>

              {/* Grid of cameras */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
                gap: '16px'
              }}>
                {network.cameras?.map((cam) => (
                  <div
                    key={cam.id}
                    style={{
                      backgroundColor: 'var(--bg-secondary)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '10px',
                      overflow: 'hidden',
                      position: 'relative',
                      display: 'flex',
                      flexDirection: 'column',
                      transition: 'transform 0.2s ease, border-color 0.2s ease'
                    }}
                  >
                    {/* Camera Thumbnail */}
                    <div style={{ position: 'relative', aspectRatio: '16/9', backgroundColor: '#000', overflow: 'hidden' }}>
                      {thumbnails[cam.id!] ? (
                        <img
                          src={thumbnails[cam.id!]}
                          alt={cam.name}
                          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        />
                      ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-secondary)' }}>
                          <Video size={32} style={{ opacity: 0.3 }} />
                          <span style={{ fontSize: '0.75rem', marginTop: '4px' }}>Chưa có preview</span>
                        </div>
                      )}

                      {/* Top Right action buttons */}
                      {canEdit && (
                        <div style={{
                          position: 'absolute',
                          top: '8px',
                          right: '8px',
                          display: 'flex',
                          gap: '6px'
                        }}>
                          <button
                            onClick={() => handleEditCamera(cam)}
                            style={{
                              backgroundColor: 'rgba(99, 102, 241, 0.9)',
                              border: 'none',
                              borderRadius: '6px',
                              width: '28px',
                              height: '28px',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              color: 'white',
                              cursor: 'pointer',
                              boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                              transition: 'background-color 0.2s'
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--accent-primary)'}
                            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'rgba(99, 102, 241, 0.9)'}
                            title="Cài đặt & Hiệu chuẩn camera"
                          >
                            <Settings size={14} />
                          </button>
                          <button
                            onClick={() => handleDeleteCamera(cam.id!)}
                            style={{
                              backgroundColor: 'rgba(239, 68, 68, 0.9)',
                              border: 'none',
                              borderRadius: '6px',
                              width: '28px',
                              height: '28px',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              color: 'white',
                              cursor: 'pointer',
                              boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                              transition: 'background-color 0.2s'
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--error)'}
                            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'rgba(239, 68, 68, 0.9)'}
                            title="Xóa camera"
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      )}
                    </div>

                    {/* Camera Body Info */}
                    <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px', flex: 1, justifyContent: 'space-between' }}>
                      <div>
                        <h5 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'white', margin: 0 }}>{cam.name}</h5>
                        {cam.location && (
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Vị trí: {cam.location}</span>
                        )}
                      </div>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {/* Calibration status */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{
                            width: '6px',
                            height: '6px',
                            borderRadius: '50%',
                            backgroundColor: cam.has_calibration ? 'var(--success)' : 'var(--warning)'
                          }} />
                          <span style={{ fontSize: '0.75rem', color: cam.has_calibration ? '#4ade80' : '#fbbf24', fontWeight: 500 }}>
                            {cam.has_calibration ? 'Đã cấu hình hiệu chuẩn' : 'Chưa cấu hình hiệu chuẩn'}
                          </span>
                        </div>

                        {/* Resolution & FPS */}
                        {cam.resolution && (
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                            {cam.resolution} @ {cam.fps} FPS
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}

                {/* Add Camera Card */}
                {canEdit && (
                  <div
                    onClick={() => setShowAddCamModal(true)}
                    style={{
                      border: '2px dashed rgba(99, 102, 241, 0.3)',
                      borderRadius: '10px',
                      aspectRatio: '16/9',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      transition: 'border-color 0.2s, background-color 0.2s, transform 0.2s',
                      backgroundColor: 'rgba(99, 102, 241, 0.02)'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = 'var(--accent-primary)';
                      e.currentTarget.style.backgroundColor = 'rgba(99, 102, 241, 0.06)';
                      e.currentTarget.style.transform = 'translateY(-2px)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = 'rgba(99, 102, 241, 0.3)';
                      e.currentTarget.style.backgroundColor = 'rgba(99, 102, 241, 0.02)';
                      e.currentTarget.style.transform = 'translateY(0)';
                    }}
                  >
                    <Plus size={32} color="var(--accent-primary)" style={{ marginBottom: '8px' }} />
                    <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>Thêm Camera Mới</span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>Upload video & hiệu chuẩn</span>
                  </div>
                )}
              </div>

              {network.cameras?.length === 0 && (
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '3rem',
                  backgroundColor: 'var(--bg-tertiary)',
                  borderRadius: '10px',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-secondary)'
                }}>
                  <Video size={40} style={{ opacity: 0.3, marginBottom: '12px' }} />
                  <p style={{ fontSize: '0.9rem', margin: 0 }}>Chưa có camera nào trong mạng lưới này.</p>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px', textAlign: 'center', maxWidth: '300px' }}>
                    Nhấn vào thẻ <strong>"Thêm Camera Mới"</strong> ở trên để upload nguồn video và cấu hình ma trận hiệu chuẩn homography.
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div style={{ width: '100%', aspectRatio: '16/9', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', padding: '2rem' }}>
              <Video size={56} style={{ marginBottom: '12px', opacity: 0.4 }} />
              <h4 style={{ color: 'white', fontWeight: 600, fontSize: '1.1rem', marginBottom: '4px' }}>
                {network.status === 'stopping' ? 'Đang đóng tiến trình...' : 'Tiến trình bám vết thất bại'}
              </h4>
              <p style={{ fontSize: '0.875rem', textAlign: 'center', maxWidth: '380px' }}>
                {network.status === 'stopping' 
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
                  backgroundColor: network.status === 'running' ? 'var(--accent-primary)' : network.status === 'completed' ? 'var(--success)' : network.status === 'failed' ? 'var(--error)' : 'var(--text-secondary)'
                }} />
                <span style={{ fontSize: '1.05rem', fontWeight: 600, textTransform: 'capitalize' }}>
                  {network.status === 'running' ? 'Đang bám vết...' : network.status === 'completed' ? 'Đã hoàn thành' : network.status === 'failed' ? 'Thất bại' : network.status === 'stopping' ? 'Đang dừng...' : 'Chưa kích hoạt'}
                </span>
              </div>

              {network.status === 'running' && (
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
                  {network.status === 'running' ? liveStats.active_globals : network.total_global_ids}
                </span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  <Clock size={16} />
                  Số Frame đã xử lý
                </span>
                <span style={{ fontSize: '1.1rem', fontWeight: 600 }}>
                  {network.status === 'running' ? liveStats.frame_id : network.total_frames}
                </span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  <Zap size={16} />
                  Tốc độ xử lý (FPS)
                </span>
                <span style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--success)' }}>
                  {network.status === 'running' ? liveStats.fps.toFixed(1) : network.avg_fps.toFixed(1)} FPS
                </span>
              </div>
            </div>
          </Card>

          {/* Download Box */}
          {network.status === 'completed' && outputVideoUrl && (
            <Card style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.5px' }}>
                Tải xuống kết quả
              </h3>
              <Button 
                onClick={async (e) => {
                  e.preventDefault();
                  try {
                    // Try fetching as blob to force download without opening a new tab
                    const response = await fetch(outputVideoUrl);
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = url;
                    a.download = `session_${networkId}_output.mp4`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    a.remove();
                  } catch (err) {
                    console.error("Fetch download failed, falling back", err);
                    // Fallback to normal anchor click
                    const a = document.createElement('a');
                    a.href = outputVideoUrl;
                    a.download = `session_${networkId}_output.mp4`;
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                  }
                }}
                style={{
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
            </Card>
          )}
        </div>
      </div>

      {/* Add Camera Modal */}
      {showAddCamModal && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.75)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1100
        }}>
          <Card style={{
            width: editingCamId ? '650px' : '500px',
            maxHeight: '90vh',
            overflowY: 'auto',
            padding: '2rem',
            position: 'relative',
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)'
          }}>
            <button 
              onClick={finishAddCamera}
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

            {uploadStep !== 'roi_drawing' ? (
              <>
                <h3 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Plus size={20} color="var(--accent-primary)" />
                  {editingCamId ? 'Cấu hình & Chỉnh sửa Camera' : 'Thêm Camera mới vào Mạng lưới'}
                </h3>

                <form onSubmit={handleSaveCamera} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Tên Camera *</label>
                <input
                  type="text"
                  placeholder="Ví dụ: Camera Cổng A, Camera Hành Lang B..."
                  value={camName}
                  onChange={(e) => setCamName(e.target.value)}
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

              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Vị trí địa lý (Không bắt buộc)</label>
                <input
                  type="text"
                  placeholder="Ví dụ: Tầng 1, Sảnh Chính..."
                  value={camLocation}
                  onChange={(e) => setCamLocation(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    color: 'white',
                    fontSize: '0.9rem'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                  Video nguồn (.mp4) {editingCamId ? '(Không bắt buộc - chọn để thay thế)' : '*'}
                </label>
                <div style={{
                  border: '1px dashed var(--border-color)',
                  borderRadius: '6px',
                  padding: '16px',
                  backgroundColor: 'var(--bg-tertiary)',
                  textAlign: 'center',
                  cursor: 'pointer',
                  position: 'relative'
                }}>
                  <input
                    type="file"
                    accept="video/mp4"
                    onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
                    style={{
                      position: 'absolute',
                      top: 0, left: 0, width: '100%', height: '100%',
                      opacity: 0, cursor: 'pointer'
                    }}
                  />
                  <Upload size={24} style={{ opacity: 0.5, marginBottom: '6px' }} />
                  <p style={{ margin: 0, fontSize: '0.85rem', fontWeight: 500 }}>
                    {videoFile ? videoFile.name : 'Chọn file video nguồn (.mp4)'}
                  </p>
                  {videoFile && (
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      {(videoFile.size / (1024 * 1024)).toFixed(2)} MB
                    </span>
                  )}
                </div>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>File hiệu chuẩn Homography (.json)</label>
                <div style={{
                  border: '1px dashed var(--border-color)',
                  borderRadius: '6px',
                  padding: '16px',
                  backgroundColor: 'var(--bg-tertiary)',
                  textAlign: 'center',
                  cursor: 'pointer',
                  position: 'relative'
                }}>
                  <input
                    type="file"
                    accept=".json"
                    onChange={(e) => setCalibFile(e.target.files?.[0] || null)}
                    style={{
                      position: 'absolute',
                      top: 0, left: 0, width: '100%', height: '100%',
                      opacity: 0, cursor: 'pointer'
                    }}
                  />
                  <Upload size={24} style={{ opacity: 0.5, marginBottom: '6px' }} />
                  <p style={{ margin: 0, fontSize: '0.85rem', fontWeight: 500 }}>
                    {calibFile ? calibFile.name : 'Chọn file JSON ma trận hiệu chuẩn'}
                  </p>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '12px', marginTop: '1rem' }}>
                <Button 
                  type="submit" 
                  disabled={uploadStep !== 'idle'}
                  style={{
                    flex: 1, height: '44px',
                    background: 'linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))',
                    border: 'none',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px'
                  }}
                >
                  {uploadStep !== 'idle' ? (
                    <>
                      <RefreshCw size={16} className="animate-spin" />
                      {uploadProgressText}
                    </>
                  ) : 'Upload và Thiết lập Camera'}
                </Button>
                <Button 
                  type="button"
                  onClick={finishAddCamera}
                  style={{ flex: 1, height: '44px', backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', color: 'white' }}
                >
                  Hủy
                </Button>
              </div>
            </form>
            </>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <h3 style={{ fontSize: '1.4rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Settings size={20} color="var(--accent-primary)" />
                  {editingCamId ? 'Chỉnh sửa Camera' : 'Bước 2: Cấu hình khu vực giám sát (ROI)'}
                </h3>

                {/* Camera info editing fields (only in edit mode) */}
                {editingCamId && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', padding: '1rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                    <div style={{ display: 'flex', gap: '12px' }}>
                      <div style={{ flex: 1 }}>
                        <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>Tên Camera</label>
                        <input
                          type="text"
                          value={camName}
                          onChange={(e) => setCamName(e.target.value)}
                          placeholder="Tên camera..."
                          style={{
                            width: '100%',
                            padding: '8px 10px',
                            backgroundColor: 'var(--bg-secondary)',
                            border: '1px solid var(--border-color)',
                            borderRadius: '6px',
                            color: 'white',
                            fontSize: '0.85rem'
                          }}
                        />
                      </div>
                      <div style={{ flex: 1 }}>
                        <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>Vị trí</label>
                        <input
                          type="text"
                          value={camLocation}
                          onChange={(e) => setCamLocation(e.target.value)}
                          placeholder="Vị trí địa lý..."
                          style={{
                            width: '100%',
                            padding: '8px 10px',
                            backgroundColor: 'var(--bg-secondary)',
                            border: '1px solid var(--border-color)',
                            borderRadius: '6px',
                            color: 'white',
                            fontSize: '0.85rem'
                          }}
                        />
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: '12px' }}>
                      <div style={{ flex: 1 }}>
                        <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                          Thay video nguồn (.mp4)
                        </label>
                        <div style={{
                          border: '1px dashed var(--border-color)',
                          borderRadius: '6px',
                          padding: '10px',
                          backgroundColor: 'var(--bg-secondary)',
                          textAlign: 'center',
                          cursor: 'pointer',
                          position: 'relative'
                        }}>
                          <input
                            type="file"
                            accept="video/mp4"
                            onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
                            style={{
                              position: 'absolute',
                              top: 0, left: 0, width: '100%', height: '100%',
                              opacity: 0, cursor: 'pointer'
                            }}
                          />
                          <p style={{ margin: 0, fontSize: '0.8rem', fontWeight: 500, color: videoFile ? 'var(--accent-secondary)' : 'var(--text-secondary)' }}>
                            {videoFile ? `✓ ${videoFile.name}` : 'Chọn file mới (không bắt buộc)'}
                          </p>
                        </div>
                      </div>
                      <div style={{ flex: 1 }}>
                        <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                          Thay file hiệu chuẩn (.json)
                        </label>
                        <div style={{
                          border: '1px dashed var(--border-color)',
                          borderRadius: '6px',
                          padding: '10px',
                          backgroundColor: 'var(--bg-secondary)',
                          textAlign: 'center',
                          cursor: 'pointer',
                          position: 'relative'
                        }}>
                          <input
                            type="file"
                            accept=".json"
                            onChange={(e) => setCalibFile(e.target.files?.[0] || null)}
                            style={{
                              position: 'absolute',
                              top: 0, left: 0, width: '100%', height: '100%',
                              opacity: 0, cursor: 'pointer'
                            }}
                          />
                          <p style={{ margin: 0, fontSize: '0.8rem', fontWeight: 500, color: calibFile ? 'var(--accent-secondary)' : 'var(--text-secondary)' }}>
                            {calibFile ? `✓ ${calibFile.name}` : 'Chọn file mới (không bắt buộc)'}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {!editingCamId && (
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Vẽ đa giác để xác định khu vực giám sát. Bạn có thể bỏ qua nếu camera này không cần ROI.
                  </p>
                )}

                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600, margin: 0 }}>
                  Khu vực giám sát (ROI)
                </p>

                <div style={{ position: 'relative', width: '100%', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-color)', marginBottom: '4px' }}>
                  <img
                    ref={imageRef}
                    src={frameUrl}
                    alt="Camera Frame"
                    style={{ width: '100%', display: 'block', minHeight: '240px', backgroundColor: 'var(--bg-tertiary)' }}
                    onError={(e) => {
                      e.currentTarget.src = "https://images.unsplash.com/photo-1557683316-973673baf926?q=80&w=1200";
                    }}
                  />
                  <svg
                    onClick={handleSvgClick}
                    viewBox={`0 0 ${getCameraResolution().width} ${getCameraResolution().height}`}
                    preserveAspectRatio="none"
                    style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', cursor: isDrawing ? 'crosshair' : 'default' }}
                  >
                    {rois.map((roi) => (
                      <polygon
                        key={roi.id} points={getPixelPoints(roi.polygon)}
                        fill="rgba(99, 102, 241, 0.15)" stroke="var(--accent-primary)" strokeWidth="4"
                      />
                    ))}
                    {currentPoints.length > 0 && (
                      <>
                        <polygon points={renderPolygonPoints(currentPoints)} fill="rgba(245, 158, 11, 0.15)" stroke="#f59e0b" strokeWidth="4" />
                        {currentPoints.map(([x, y], idx) => (
                          <circle key={idx} cx={x * getCameraResolution().width} cy={y * getCameraResolution().height} r="10" fill="#f59e0b" stroke="white" strokeWidth="2" />
                        ))}
                      </>
                    )}
                  </svg>
                </div>

                {!isDrawing ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <Button onClick={startDrawingWorkflow} variant="secondary" style={{ width: '100%', height: '40px', fontWeight: 600 }}>
                      + Thêm ROI mới
                    </Button>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <p style={{ fontSize: '0.85rem', color: 'var(--accent-secondary)', margin: 0, fontWeight: 500 }}>
                      Đang vẽ vùng: <strong style={{ color: 'white' }}>"{newRoiName}"</strong>. Nhấp chuột lên hình ảnh để chọn các điểm đỉnh cho đa giác (cần tối thiểu 3 điểm).
                    </p>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <Button onClick={handleAddRoi} variant="primary" style={{ flex: 1, height: '36px', fontSize: '0.85rem' }}>
                        Hoàn tất vẽ
                      </Button>
                      {currentPoints.length > 0 && (
                        <Button onClick={() => setCurrentPoints([])} variant="secondary" style={{ height: '36px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                          Vẽ lại
                        </Button>
                      )}
                      <Button onClick={() => { setIsDrawing(false); setCurrentPoints([]); }} variant="secondary" style={{ height: '36px', fontSize: '0.85rem', color: 'var(--error)' }}>
                        Hủy
                      </Button>
                    </div>
                  </div>
                )}

                {rois.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '120px', overflowY: 'auto' }}>
                    {rois.map((roi) => (
                      <div key={roi.id} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 12px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '6px' }}>
                        <span style={{ fontSize: '0.85rem', color: 'white' }}>{roi.name}</span>
                        <button onClick={() => handleDeleteRoi(roi.id)} style={{ background: 'transparent', border: 'none', color: 'var(--error)', cursor: 'pointer' }}>
                          <Trash2 size={14} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                <div style={{ display: 'flex', gap: '12px', marginTop: '1rem' }}>
                  <Button onClick={handleSaveRois} style={{ flex: 1, background: 'linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))' }}>
                    Lưu cấu hình & Hoàn tất
                  </Button>
                  <Button onClick={finishAddCamera} style={{ flex: 1, backgroundColor: 'var(--bg-tertiary)' }}>
                    Bỏ qua & Hoàn tất
                  </Button>
                </div>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Uploading loading overlay */}
      {uploadStep !== 'idle' && uploadStep !== 'roi_drawing' && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(10, 11, 14, 0.85)',
          backdropFilter: 'blur(8px)',
          zIndex: 2000,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          padding: '2rem'
        }}>
          <div style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '2.5rem',
            maxWidth: '450px',
            width: '100%',
            textAlign: 'center',
            boxShadow: '0 25px 50px -12px rgba(0,0,0,0.5)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '20px'
          }}>
            <div style={{ position: 'relative', width: '60px', height: '60px' }}>
              <Loader2 className="animate-spin" size={60} color="var(--accent-primary)" style={{ animationDuration: '1.5s' }} />
              <div style={{
                position: 'absolute',
                top: 0, left: 0, right: 0, bottom: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <Upload size={22} color="var(--accent-secondary)" />
              </div>
            </div>

            <div>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '8px' }}>Đang nạp dữ liệu camera</h3>
              <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', lineHeight: '1.5', minHeight: '44px', margin: 0 }}>
                {uploadProgressText}
              </p>
            </div>

            {/* Stepper view */}
            <div style={{ display: 'flex', alignItems: 'center', width: '100%', gap: '8px', marginTop: '10px' }}>
              <div style={{ flex: 1, height: '4px', borderRadius: '2px', backgroundColor: 'var(--accent-primary)' }} />
              <div style={{
                flex: 1,
                height: '4px',
                borderRadius: '2px',
                backgroundColor: ['uploading_video', 'uploading_calib', 'finishing'].includes(uploadStep) ? 'var(--accent-primary)' : 'var(--bg-tertiary)'
              }} />
              <div style={{
                flex: 1,
                height: '4px',
                borderRadius: '2px',
                backgroundColor: ['uploading_calib', 'finishing'].includes(uploadStep) ? 'var(--accent-primary)' : 'var(--bg-tertiary)'
              }} />
              <div style={{
                flex: 1,
                height: '4px',
                borderRadius: '2px',
                backgroundColor: uploadStep === 'finishing' ? 'var(--accent-primary)' : 'var(--bg-tertiary)'
              }} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
