import { useEffect, useState, useRef } from 'react';
import { Card } from '../components/Common/Card';
import { Button } from '../components/Common/Button';
import { Input } from '../components/Common/Input';
import { CameraService, CameraNetworkService, CameraRoiService } from '../api/services';
import type { Camera, CameraNetwork } from '../types';
import { Plus, Trash2, Video, FileCode, CheckCircle, AlertTriangle, Upload, Settings, MapPin } from 'lucide-react';

interface RoiConfig {
  id: string;
  name: string;
  polygon: number[][]; // [[x1, y1], [x2, y2], ...] in pixel space
}

export function Cameras() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [networks, setNetworks] = useState<CameraNetwork[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  // ROI states
  const [rois, setRois] = useState<RoiConfig[]>([]);
  const [frameUrl, setFrameUrl] = useState<string>('');
  const [currentPoints, setCurrentPoints] = useState<number[][]>([]); // normalized coordinates [[nx, ny], ...]
  const [newRoiName, setNewRoiName] = useState('');
  const imageRef = useRef<HTMLImageElement>(null);
  
  // Forms
  const [newCamera, setNewCamera] = useState({ name: '', source: '', location: '', network_id: '' as number | '' });
  const [selectedCalibFile, setSelectedCalibFile] = useState<File | null>(null);
  const [selectedVideoFile, setSelectedVideoFile] = useState<File | null>(null);
  const [thumbnailUrls, setThumbnailUrls] = useState<Record<number, string>>({});
  
  const [loading, setLoading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  
  // Connection Check & Video Flow States
  const [sourceType, setSourceType] = useState<'rtsp' | 'video'>('rtsp');
  const [connectionStatus, setConnectionStatus] = useState<null | 'loading' | 'success' | 'failed'>(null);
  const [connectionMsg, setConnectionMsg] = useState('');
  const [modalVideoFile, setModalVideoFile] = useState<File | null>(null);

  const userRole = localStorage.getItem('user_role');
  const canEdit = userRole === 'ADMIN' || userRole === 'OPERATOR';

  useEffect(() => {
    loadCameras();
  }, []);

  const loadCameras = async () => {
    try {
      const data = await CameraService.getAll();
      const nets = await CameraNetworkService.getAll();
      setNetworks(nets);
      setCameras(data);
      if (data.length > 0) {
        // Fetch thumbnails for all cameras
        const thumbs: Record<number, string> = {};
        for (const cam of data) {
          if (cam.id) {
            try {
              const res = await CameraService.getThumbnail(cam.id);
              thumbs[cam.id] = res.url;
            } catch (e) {
              // No thumbnail yet
            }
          }
        }
        setThumbnailUrls(thumbs);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleCheckConnection = async () => {
    if (!newCamera.source) {
      setConnectionStatus('failed');
      setConnectionMsg('Vui lòng nhập nguồn camera trước!');
      return;
    }
    setConnectionStatus('loading');
    setConnectionMsg('');
    try {
      const res = await CameraService.checkConnection(newCamera.source, 'rtsp');
      if (res.ok) {
        setConnectionStatus('success');
        setConnectionMsg(`Kết nối thành công! Độ phân giải: ${res.resolution} (${res.fps} FPS)`);
      } else {
        setConnectionStatus('failed');
        setConnectionMsg(res.message || 'Không thể kết nối đến camera.');
      }
    } catch (err: any) {
      setConnectionStatus('failed');
      setConnectionMsg(err.response?.data?.detail || 'Lỗi khi kiểm tra kết nối.');
    }
  };

  const handleAdd = async () => {
    if (!newCamera.name) return;
    if (sourceType === 'rtsp' && !newCamera.source) return;
    if (sourceType === 'video' && !modalVideoFile) return;

    setLoading(true);
    try {
      if (sourceType === 'video' && modalVideoFile) {
        // Create camera first with pending video path
        const cam = await CameraService.create({
          name: newCamera.name,
          source: `videos/${modalVideoFile.name}`,
          location: newCamera.location,
          network_id: newCamera.network_id ? Number(newCamera.network_id) : undefined,
          is_active: true
        });
        
        if (cam.id) {
          await CameraService.uploadVideo(cam.id, modalVideoFile);
        }
      } else {
        await CameraService.create({
          name: newCamera.name,
          source: newCamera.source,
          location: newCamera.location,
          network_id: newCamera.network_id ? Number(newCamera.network_id) : undefined,
          is_active: true
        });
      }

      setIsModalOpen(false);
      setNewCamera({ name: '', source: '', location: '', network_id: '' });
      setModalVideoFile(null);
      setConnectionStatus(null);
      setConnectionMsg('');
      loadCameras();
    } catch (e) {
      console.error(e);
      alert('Lỗi khi thêm camera. Vui lòng kiểm tra kết nối/tập tin video.');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm('Bạn có chắc chắn muốn xóa camera này? Tất cả phiên bám vết liên quan có thể bị ảnh hưởng.')) return;
    try {
      await CameraService.delete(id);
      setSelectedCamera(null);
      loadCameras();
    } catch (e) {
      console.error(e);
    }
  };

  const handleCalibrationUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedCamera || !selectedCamera.id || !selectedCalibFile) return;

    setLoading(true);
    setUploadError(null);
    setSuccessMsg(null);
    try {
      await CameraService.uploadCalibration(selectedCamera.id, selectedCalibFile);
      setSuccessMsg('Tải lên cấu hình hiệu chuẩn thành công!');
      setSelectedCalibFile(null);
      // Reload camera details
      const updatedCam = await CameraService.getAll();
      const match = updatedCam.find(c => c.id === selectedCamera.id);
      if (match) setSelectedCamera(match);
      loadCameras();
    } catch (err: any) {
      setUploadError(err.response?.data?.detail || 'Lỗi tải lên hiệu chuẩn');
    } finally {
      setLoading(false);
    }
  };

  const handleVideoUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedCamera || !selectedCamera.id || !selectedVideoFile) return;

    setLoading(true);
    setUploadError(null);
    setSuccessMsg(null);
    try {
      await CameraService.uploadVideo(selectedCamera.id, selectedVideoFile);
      setSuccessMsg('Tải lên video camera mẫu thành công! Hệ thống đã trích xuất thumbnail.');
      setSelectedVideoFile(null);
      // Reload camera details
      const updatedCam = await CameraService.getAll();
      const match = updatedCam.find(c => c.id === selectedCamera.id);
      if (match) setSelectedCamera(match);
      loadCameras();
    } catch (err: any) {
      setUploadError(err.response?.data?.detail || 'Lỗi tải lên video');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedCamera?.id) {
      const cameraId = selectedCamera.id;
      CameraRoiService.getRois(cameraId)
        .then(setRois)
        .catch(err => console.error('Error getting ROIs:', err));
      setCurrentPoints([]);
      setNewRoiName('');

      const loadFrame = async () => {
        try {
          const { apiClient } = await import('../api/client');
          const response = await apiClient.get(`/cameras/${cameraId}/frame`, { responseType: 'blob' });
          const url = URL.createObjectURL(response.data);
          setFrameUrl(url);
        } catch (err) {
          console.error('Error loading camera frame:', err);
          setFrameUrl(''); // fallback to error image
        }
      };
      loadFrame();
    } else {
      setRois([]);
      setFrameUrl('');
      setCurrentPoints([]);
    }
  }, [selectedCamera]);

  // Convert SVG click to normalized point (0.0 to 1.0)
  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
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
    
    if (!selectedCamera) return;
    
    // Parse resolution safely to convert normalized to pixel space
    let width = 1920;
    let height = 1080;
    if (selectedCamera.resolution) {
      const parts = selectedCamera.resolution.split('x').map(Number);
      if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1]) && parts[0] > 0 && parts[1] > 0) {
        width = parts[0];
        height = parts[1];
      }
    }
    
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
  };

  const handleSaveRois = async () => {
    if (!selectedCamera?.id) return;
    try {
      await CameraRoiService.saveRois(selectedCamera.id, rois);
      alert('Đã lưu cấu hình các khu vực ROI thành công!');
    } catch (err) {
      console.error(err);
      alert('Không thể lưu cấu hình ROI.');
    }
  };

  const handleDeleteRoi = (id: string) => {
    setRois(rois.filter(r => r.id !== id));
  };

  // Render polygon points on SVG
  const renderPolygonPoints = (pts: number[][]) => {
    return pts.map(([nx, ny]) => `${nx * 100}%,${ny * 100}%`).join(' ');
  };

  // Convert pixel-space polygon to percentage-space for frontend rendering
  const getPercentagePoints = (pixelPoly: number[][]) => {
    if (!pixelPoly) return '';
    let width = 1920;
    let height = 1080;
    if (selectedCamera?.resolution) {
      const parts = selectedCamera.resolution.split('x').map(Number);
      if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1]) && parts[0] > 0 && parts[1] > 0) {
        width = parts[0];
        height = parts[1];
      }
    }
    
    return pixelPoly.map(([px, py]) => `${(px / width) * 100}%, ${(py / height) * 100}%`).join(' ');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-lg)' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.5px' }}>Quản lý Camera</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Thêm camera giám sát và hiệu chuẩn ma trận đồng dạng homography</p>
        </div>
        {canEdit && (
          <Button onClick={() => setIsModalOpen(true)} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Plus size={18} />
            Thêm Camera mới
          </Button>
        )}
      </div>

      {/* Main Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 'var(--spacing-lg)' }}>
        {/* Left: Camera List */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', alignContent: 'flex-start' }}>
          {cameras.map(cam => (
            <Card 
              key={cam.id}
              onClick={() => { setSelectedCamera(cam); setUploadError(null); setSuccessMsg(null); }}
              style={{
                padding: 'var(--spacing-md)',
                cursor: 'pointer',
                borderColor: selectedCamera?.id === cam.id ? 'var(--accent-primary)' : 'var(--border-color)',
                backgroundColor: selectedCamera?.id === cam.id ? 'var(--bg-secondary)' : 'var(--bg-secondary)',
                display: 'flex',
                flexDirection: 'column',
                gap: '12px',
                transition: 'all 0.2s ease'
              }}
            >
              <div style={{ display: 'flex', justifyItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Video size={18} color="var(--accent-primary)" />
                  <span style={{ fontWeight: 600, fontSize: '1rem' }}>{cam.name}</span>
                </div>
                {cam.has_calibration ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.75rem', color: 'var(--success)' }}>
                    <CheckCircle size={14} />
                    Calibrated
                  </span>
                ) : (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.75rem', color: '#f59e0b' }}>
                    <AlertTriangle size={14} />
                    Uncalibrated
                  </span>
                )}
              </div>

              {/* Preview Thumbnail or Stream */}
              <div style={{ width: '100%', aspectRatio: '16/9', borderRadius: '8px', overflow: 'hidden', backgroundColor: 'black', position: 'relative' }}>
                {thumbnailUrls[cam.id!] ? (
                  <img 
                    src={thumbnailUrls[cam.id!]} 
                    alt={cam.name} 
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                    Chưa có ảnh preview
                  </div>
                )}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                <MapPin size={12} />
                <span>{cam.location || 'Chưa định vị'}</span>
                {cam.resolution && <span>• {cam.resolution} ({cam.fps} FPS)</span>}
              </div>
            </Card>
          ))}
        </div>

        {/* Right: Camera Details, Calibration & Video Uploads */}
        <Card style={{ padding: 'var(--spacing-xl)' }}>
          {selectedCamera ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px' }}>
                <div>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Cấu hình chi tiết</h3>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Camera ID: #{selectedCamera.id}</span>
                </div>
                {userRole === 'ADMIN' && (
                  <Button 
                    onClick={() => handleDelete(selectedCamera.id!)}
                    style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', color: 'var(--error)', display: 'flex', alignItems: 'center', gap: '4px' }}
                  >
                    <Trash2 size={16} />
                    Xóa Camera
                  </Button>
                )}
              </div>

              {uploadError && (
                <div style={{ padding: '0.75rem 1rem', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', color: 'var(--error)', fontSize: '0.85rem' }}>
                  {uploadError}
                </div>
              )}
              {successMsg && (
                <div style={{ padding: '0.75rem 1rem', backgroundColor: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)', borderRadius: '8px', color: 'var(--success)', fontSize: '0.85rem' }}>
                  {successMsg}
                </div>
              )}

              {/* Metadata */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '0.9rem' }}>
                <div>
                  <span style={{ color: 'var(--text-secondary)', display: 'block', fontSize: '0.8rem' }}>Tên Camera</span>
                  <strong>{selectedCamera.name}</strong>
                </div>
                <div>
                  <span style={{ color: 'var(--text-secondary)', display: 'block', fontSize: '0.8rem' }}>Vị trí đặt</span>
                  <strong>{selectedCamera.location || 'Chưa định vị'}</strong>
                </div>
                <div style={{ gridColumn: 'span 2' }}>
                  <span style={{ color: 'var(--text-secondary)', display: 'block', fontSize: '0.8rem' }}>Nguồn camera / Video source</span>
                  <code style={{ fontSize: '0.8rem', backgroundColor: 'var(--bg-primary)', padding: '2px 6px', borderRadius: '4px', display: 'block', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {selectedCamera.source}
                  </code>
                </div>
              </div>

              {/* ROI Configuration Section */}
              <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1.25rem' }}>
                <h4 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Plus size={18} color="var(--accent-primary)" />
                  Cấu hình khu vực giám sát (ROI)
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '12px' }}>
                  Vẽ đa giác trên ảnh camera dưới đây để xác định khu vực cần đếm người và đo thời gian tập trung. Nhấp chuột vào hình để vẽ đa giác.
                </p>

                {/* Draw Canvas */}
                <div style={{ position: 'relative', width: '100%', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-color)', marginBottom: '12px' }}>
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
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '100%',
                      cursor: 'crosshair'
                    }}
                  >
                    {/* Render saved ROIs */}
                    {rois.map((roi) => (
                      <polygon
                        key={roi.id}
                        points={getPercentagePoints(roi.polygon)}
                        fill="rgba(99, 102, 241, 0.15)"
                        stroke="var(--accent-primary)"
                        strokeWidth="2"
                      />
                    ))}

                    {/* Render currently drawing polygon */}
                    {currentPoints.length > 0 && (
                      <>
                        <polygon
                          points={renderPolygonPoints(currentPoints)}
                          fill="rgba(245, 158, 11, 0.15)"
                          stroke="#f59e0b"
                          strokeWidth="2"
                        />
                        {currentPoints.map(([x, y], idx) => (
                          <circle
                            key={idx}
                            cx={`${x * 100}%`}
                            cy={`${y * 100}%`}
                            r="5"
                            fill="#f59e0b"
                            stroke="white"
                            strokeWidth="1"
                          />
                        ))}
                      </>
                    )}
                  </svg>
                </div>

                <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                  <Input
                    type="text"
                    placeholder="Tên khu vực (ví dụ: Cửa chính)"
                    value={newRoiName}
                    onChange={(e) => setNewRoiName(e.target.value)}
                    style={{ flex: 1, height: '36px', fontSize: '0.85rem' }}
                  />
                  <Button
                    onClick={handleAddRoi}
                    variant="secondary"
                    style={{ height: '36px', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '4px' }}
                  >
                    <Plus size={14} />
                    Thêm
                  </Button>
                  {currentPoints.length > 0 && (
                    <Button
                      onClick={() => setCurrentPoints([])}
                      variant="secondary"
                      style={{ height: '36px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}
                    >
                      Vẽ lại
                    </Button>
                  )}
                </div>

                {/* ROI list */}
                {rois.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px', maxHeight: '150px', overflowY: 'auto' }}>
                    {rois.map((roi) => (
                      <div key={roi.id} style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '6px 12px',
                        backgroundColor: 'var(--bg-tertiary)',
                        borderRadius: '6px',
                        border: '1px solid var(--border-color)'
                      }}>
                        <span style={{ fontSize: '0.85rem', color: 'white', fontWeight: 500 }}>{roi.name}</span>
                        <button
                          onClick={() => handleDeleteRoi(roi.id)}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: 'var(--error)',
                            cursor: 'pointer',
                            padding: '2px'
                          }}
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                <Button
                  onClick={handleSaveRois}
                  style={{ width: '100%', height: '38px', fontSize: '0.85rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
                >
                  Lưu Cấu Hình ROI
                </Button>
              </div>

              {/* Action 1: Upload Calibration JSON */}
              <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1.25rem' }}>
                <h4 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <FileCode size={18} color="var(--accent-primary)" />
                  Hiệu chuẩn ma trận đồng dạng (Homography)
                </h4>
                
                {selectedCamera.has_calibration ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--success)', fontSize: '0.85rem', marginBottom: '12px' }}>
                    <CheckCircle size={16} />
                    <span>Camera đã được nạp file hiệu chuẩn thành công! ({selectedCamera.calibration_path})</span>
                  </div>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#f59e0b', fontSize: '0.85rem', marginBottom: '12px' }}>
                    <AlertTriangle size={16} />
                    <span>Camera chưa được hiệu chuẩn. Luồng multicam cần file hiệu chuẩn này để ánh xạ tọa độ 3D!</span>
                  </div>
                )}

                {canEdit && (
                  <form onSubmit={handleCalibrationUpload} style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <input 
                      type="file" 
                      accept=".json"
                      onChange={(e) => setSelectedCalibFile(e.target.files ? e.target.files[0] : null)}
                      style={{ fontSize: '0.85rem', flex: 1 }}
                      required
                    />
                    <Button 
                      type="submit" 
                      disabled={loading || !selectedCalibFile}
                      style={{ height: '36px', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '4px' }}
                    >
                      <Upload size={14} />
                      Tải lên JSON
                    </Button>
                  </form>
                )}
              </div>

              {/* Action 2: Upload Video File */}
              <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1.25rem' }}>
                <h4 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Video size={18} color="var(--accent-secondary)" />
                  Tải lên Video nguồn Camera
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '12px' }}>
                  Upload file video mp4 của camera lên hệ thống lưu trữ MinIO để sử dụng làm nguồn bám vết mẫu.
                </p>

                {canEdit && (
                  <form onSubmit={handleVideoUpload} style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <input 
                      type="file" 
                      accept="video/*"
                      onChange={(e) => setSelectedVideoFile(e.target.files ? e.target.files[0] : null)}
                      style={{ fontSize: '0.85rem', flex: 1 }}
                      required
                    />
                    <Button 
                      type="submit" 
                      disabled={loading || !selectedVideoFile}
                      style={{ height: '36px', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '4px' }}
                    >
                      <Upload size={14} />
                      Tải lên Video
                    </Button>
                  </form>
                )}
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '300px', color: 'var(--text-secondary)' }}>
              <Settings size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
              <p>Chọn một Camera ở danh sách bên trái để xem hiệu chuẩn và cấu hình</p>
            </div>
          )}
        </Card>
      </div>

      {/* Creation Modal */}
      {isModalOpen && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', 
          backdropFilter: 'blur(3px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1100
        }}>
          <Card style={{ width: '480px', padding: '2rem', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Thêm Camera mới</h3>
            
            {/* Source Type Tabs */}
            <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px' }}>
              <button 
                type="button"
                onClick={() => { setSourceType('rtsp'); setConnectionStatus(null); setConnectionMsg(''); }}
                style={{
                  flex: 1, padding: '8px', borderRadius: '6px', fontSize: '0.85rem', fontWeight: 600, border: '1px solid var(--border-color)',
                  backgroundColor: sourceType === 'rtsp' ? 'var(--accent-primary)' : 'transparent',
                  color: sourceType === 'rtsp' ? 'black' : 'var(--text-secondary)',
                  cursor: 'pointer', transition: 'all 0.2s ease'
                }}
              >
                RTSP / Webcam / Local Path
              </button>
              <button 
                type="button"
                onClick={() => { setSourceType('video'); setConnectionStatus(null); setConnectionMsg(''); }}
                style={{
                  flex: 1, padding: '8px', borderRadius: '6px', fontSize: '0.85rem', fontWeight: 600, border: '1px solid var(--border-color)',
                  backgroundColor: sourceType === 'video' ? 'var(--accent-primary)' : 'transparent',
                  color: sourceType === 'video' ? 'black' : 'var(--text-secondary)',
                  cursor: 'pointer', transition: 'all 0.2s ease'
                }}
              >
                Tải lên video trực tiếp
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>Tên Camera</label>
                <Input 
                  placeholder="Ví dụ: Camera Cửa Ra Vào" 
                  value={newCamera.name}
                  onChange={e => setNewCamera({...newCamera, name: e.target.value})}
                  style={{ width: '100%' }}
                />
              </div>

              {sourceType === 'rtsp' ? (
                <div>
                  <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>Nguồn URL / RTSP stream / webcam ID / file video cục bộ</label>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <Input 
                      placeholder="Ví dụ: rtsp://192.168.1.100:554/stream1 hoặc 0 hoặc mct_demo.mp4" 
                      value={newCamera.source}
                      onChange={e => setNewCamera({...newCamera, source: e.target.value})}
                      style={{ flex: 1 }}
                    />
                    <Button 
                      onClick={handleCheckConnection} 
                      disabled={loading || connectionStatus === 'loading'}
                      style={{ padding: '0 12px', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '4px', whiteSpace: 'nowrap' }}
                    >
                      {connectionStatus === 'loading' ? 'Đang thử...' : 'Check Connection'}
                    </Button>
                  </div>
                  {connectionStatus && (
                    <div style={{
                      marginTop: '8px', fontSize: '0.8rem', padding: '8px 10px', borderRadius: '6px',
                      backgroundColor: connectionStatus === 'success' ? 'rgba(34,197,94,0.1)' : connectionStatus === 'failed' ? 'rgba(239,68,68,0.1)' : 'rgba(255,255,255,0.05)',
                      border: connectionStatus === 'success' ? '1px solid var(--success)' : connectionStatus === 'failed' ? '1px solid var(--error)' : '1px solid var(--border-color)',
                      color: connectionStatus === 'success' ? 'var(--success)' : connectionStatus === 'failed' ? 'var(--error)' : 'var(--text-secondary)',
                      lineHeight: '1.3'
                    }}>
                      {connectionMsg}
                    </div>
                  )}
                </div>
              ) : (
                <div>
                  <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>Chọn tập tin video MP4 nguồn camera</label>
                  <input 
                    type="file" 
                    accept="video/*"
                    onChange={e => setModalVideoFile(e.target.files ? e.target.files[0] : null)}
                    style={{ width: '100%', fontSize: '0.85rem', padding: '6px 0', color: 'white' }}
                  />
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: '1.3' }}>
                    Hệ thống sẽ tự động tạo camera và đẩy tập tin video này lên MinIO. Sau khi tải lên, hệ thống sẽ tự động giải mã thông số và trích xuất ảnh thumbnail.
                  </p>
                </div>
              )}

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>Vị trí đặt camera</label>
                <Input 
                  placeholder="Ví dụ: Tầng 1 - Sảnh A" 
                  value={newCamera.location}
                  onChange={e => setNewCamera({...newCamera, location: e.target.value})}
                  style={{ width: '100%' }}
                />
              </div>


              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>Chọn Camera Network</label>
                <select 
                  value={newCamera.network_id} 
                  onChange={e => setNewCamera({...newCamera, network_id: e.target.value ? Number(e.target.value) : ''})}
                  style={{ width: '100%', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-tertiary)', color: 'white' }}
                >
                  <option value="">-- Không thuộc Network nào --</option>
                  {networks.map(n => <option key={n.id} value={n.id}>{n.name}</option>)}
                </select>
              </div>

              <div style={{ display: 'flex', gap: '12px', marginTop: '10px' }}>

                <Button 
                  onClick={handleAdd} 
                  disabled={loading || (sourceType === 'rtsp' && !newCamera.source) || (sourceType === 'video' && !modalVideoFile)} 
                  style={{ flex: 1 }}
                >
                  {loading ? 'Đang xử lý...' : 'Thêm'}
                </Button>
                <Button 
                  variant="secondary" 
                  onClick={() => { setIsModalOpen(false); setModalVideoFile(null); setConnectionStatus(null); setConnectionMsg(''); }} 
                  style={{ flex: 1, backgroundColor: 'var(--bg-tertiary)', color: 'white' }}
                >
                  Hủy
                </Button>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
