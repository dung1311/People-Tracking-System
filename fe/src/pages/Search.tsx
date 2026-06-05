import { useState, useRef, useEffect } from 'react';
import { Card } from '../components/Common/Card';
import { Button } from '../components/Common/Button';
import { SearchService, CameraNetworkService } from '../api/services';
import type { SearchResult, CameraNetwork } from '../api/services';
import {
  Upload,
  Search as SearchIcon,
  Clock,
  AlertTriangle,
  MapPin,
  Camera as CameraIcon,
  RefreshCw,
  X,
  FileImage
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000/api/v1';

// Helper component to display match with bbox
function MatchItem({ group }: { group: SearchResult['matches'][0] }) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [scale, setScale] = useState(1);
  const [isHovered, setIsHovered] = useState(false);

  const handleLoad = () => {
    if (imgRef.current) {
      const { naturalWidth, clientWidth } = imgRef.current;
      setScale(clientWidth / naturalWidth);
    }
  };

  // Update scale on window resize
  useEffect(() => {
    const handleResize = () => {
      if (imgRef.current) {
        const { naturalWidth, clientWidth } = imgRef.current;
        setScale(clientWidth / naturalWidth);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const { best_match, count, start_time, end_time, camera_id } = group;
  const [x1, y1, x2, y2] = best_match.bbox;

  // Formatting date and time
  const formatTime = (iso: string) => {
    try {
      const date = new Date(iso);
      return date.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return iso;
    }
  };

  const formatDate = (iso: string) => {
    try {
      const date = new Date(iso);
      return date.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric' });
    } catch {
      return '';
    }
  };

  // Similarity calculation (Cosine distance is [0, 2], where 0 is identical)
  const similarity = Math.max(0, (1 - best_match.distance) * 100);

  // Confidence styling details
  let confidenceColor = 'var(--error)';
  let confidenceLabel = 'Độ tin cậy Thấp';
  let confidenceBg = 'rgba(239, 68, 68, 0.1)';

  if (similarity >= 75) {
    confidenceColor = 'var(--success)';
    confidenceLabel = 'Độ tin cậy Cao';
    confidenceBg = 'rgba(34, 197, 94, 0.1)';
  } else if (similarity >= 50) {
    confidenceColor = '#f59e0b'; // Amber
    confidenceLabel = 'Độ tin cậy Trung bình';
    confidenceBg = 'rgba(245, 158, 11, 0.1)';
  }

  return (
    <div
      style={{
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: 'var(--radius-lg)',
        border: isHovered ? '1px solid var(--accent-primary)' : '1px solid var(--border-color)',
        padding: 'var(--spacing-md)',
        boxShadow: isHovered
          ? '0 12px 20px -8px rgba(99, 102, 241, 0.25), 0 4px 6px -2px rgba(0, 0, 0, 0.3)'
          : '0 4px 6px -1px rgba(0, 0, 0, 0.2)',
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        position: 'relative',
        overflow: 'hidden'
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Top Indicator Accent Line */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '4px',
        background: `linear-gradient(90deg, ${confidenceColor}, var(--accent-primary))`,
        opacity: isHovered ? 1 : 0.6,
        transition: 'opacity 0.2s ease'
      }} />

      {/* Bounding Box Image Preview */}
      <div style={{
        position: 'relative',
        width: '100%',
        overflow: 'hidden',
        borderRadius: 'var(--radius-md)',
        backgroundColor: '#0a0b0d',
        border: '1px solid var(--border-color)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <img
          ref={imgRef}
          src={`${API_BASE_URL}/frames/${camera_id}/${best_match.frame_id}`}
          alt={`Global ID #${group.person_id}`}
          style={{ width: '100%', height: 'auto', display: 'block' }}
          onLoad={handleLoad}
        />

        {/* Drawn Bounding Box */}
        <div
          style={{
            position: 'absolute',
            left: x1 * scale,
            top: y1 * scale,
            width: (x2 - x1) * scale,
            height: (y2 - y1) * scale,
            border: `2.5px dashed ${confidenceColor}`,
            boxShadow: `0 0 12px ${confidenceColor}, inset 0 0 6px ${confidenceColor}`,
            backgroundColor: 'rgba(99, 102, 241, 0.05)',
            pointerEvents: 'none',
            borderRadius: '2px',
            transition: 'all 0.2s ease'
          }}
        >
          {/* Bounding Box Label tag */}
          <div style={{
            position: 'absolute',
            top: '-22px',
            left: '-2.5px',
            background: confidenceColor,
            color: 'white',
            fontSize: '10px',
            padding: '2px 6px',
            fontWeight: 'bold',
            borderRadius: '3px 3px 0 0',
            whiteSpace: 'nowrap',
            boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
          }}>
            {similarity.toFixed(1)}%
          </div>
        </div>

        {/* Info badge Overlay on image */}
        <div style={{
          position: 'absolute',
          bottom: '8px',
          left: '8px',
          backgroundColor: 'rgba(15, 17, 21, 0.75)',
          backdropFilter: 'blur(4px)',
          padding: '4px 8px',
          borderRadius: '4px',
          fontSize: '0.75rem',
          border: '1px solid rgba(255,255,255,0.1)',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}>
          <CameraIcon size={12} color="var(--accent-primary)" />
          <span style={{ color: 'white', fontWeight: 500 }}>Frame {best_match.frame_id}</span>
        </div>
      </div>

      {/* Info details */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', flexGrow: 1 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{
              fontSize: '1.25rem',
              fontWeight: 800,
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              Global ID #{group.person_id}
              <span style={{
                fontSize: '0.7rem',
                fontWeight: 700,
                color: confidenceColor,
                backgroundColor: confidenceBg,
                padding: '2px 8px',
                borderRadius: '12px',
                border: `1px solid ${confidenceColor}33`,
                letterSpacing: '0.5px'
              }}>
                {similarity.toFixed(0)}% Match • {confidenceLabel}
              </span>
            </div>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              marginTop: '4px',
              color: 'var(--text-secondary)',
              fontSize: '0.85rem'
            }}>
              <MapPin size={14} color="var(--accent-primary)" />
              <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                Camera {camera_id}
              </span>
            </div>
          </div>
        </div>

        <div style={{
          backgroundColor: 'var(--bg-tertiary)',
          borderRadius: 'var(--radius-md)',
          padding: '8px 12px',
          marginTop: '4px',
          border: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          gap: '6px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Thời gian xuất hiện:</span>
            <span style={{ color: 'white', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Clock size={12} color="var(--accent-secondary)" />
              {formatTime(start_time)} - {formatTime(end_time)}
            </span>
          </div>
          {formatDate(start_time) && (
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Ngày phát hiện:</span>
              <span style={{ color: 'white', fontWeight: 500 }}>{formatDate(start_time)}</span>
            </div>
          )}
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Số lượt ghi nhận:</span>
            <span style={{
              color: 'var(--accent-primary)',
              fontWeight: 700
            }}>
              {count} frames
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function Search() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult['matches']>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  // Camera Network Selection state
  const [networks, setNetworks] = useState<CameraNetwork[]>([]);
  const [selectedNetworkId, setSelectedNetworkId] = useState<number | ''>('');
  const [loadingNetworks, setLoadingNetworks] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Fetch camera networks on mount
  useEffect(() => {
    const fetchNetworks = async () => {
      setLoadingNetworks(true);
      try {
        const data = await CameraNetworkService.getAll();
        setNetworks(data);
        if (data.length === 1 && data[0].id) {
          setSelectedNetworkId(data[0].id);
        }
      } catch (err) {
        console.error("Failed to load camera networks", err);
      } finally {
        setLoadingNetworks(false);
      }
    };
    fetchNetworks();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const f = e.target.files[0];
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setValidationError(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const f = e.dataTransfer.files[0];
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setValidationError(null);
    }
  };

  const handleClearImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    setFile(null);
    setPreview(null);
  };

  const handleSearch = async () => {
    if (!selectedNetworkId) {
      setValidationError('Vui lòng chọn một Camera Network trước khi tìm kiếm.');
      return;
    }
    if (!file) {
      setValidationError('Vui lòng tải lên hoặc kéo thả ảnh mục tiêu.');
      return;
    }

    setValidationError(null);
    setLoading(true);
    setResults([]);
    setHasSearched(true);

    try {
      const data = await SearchService.searchPromise(file, selectedNetworkId);
      setResults(data.matches);
    } catch (e) {
      alert('Không thể thực hiện tìm kiếm đối tượng. Vui lòng kiểm tra lại kết nối backend hoặc tệp tin.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-xl)' }}>

      {/* Page Header */}
      <div>
        <h2 style={{
          fontSize: '2.25rem',
          fontWeight: 800,
          letterSpacing: '-0.75px',
          background: 'linear-gradient(135deg, white 60%, var(--text-secondary))',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: '6px'
        }}>
          Tìm kiếm đối tượng (Person Search)
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', maxWidth: '800px' }}>
          Đổi mới giao diện ReID: Nhận diện và định vị đối tượng trên toàn bộ camera trong mạng lưới dựa trên ảnh tải lên. Tìm ra Global ID đồng nhất trong Camera Network được chọn.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 'var(--spacing-xl)', alignItems: 'start' }}>

        {/* Control Box: Selector + Dropzone */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-lg)' }}>
          <Card title="Cấu hình & Tải ảnh">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '8px' }}>

              {/* 1. Camera Network Dropdown */}
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  color: 'white',
                  marginBottom: '8px'
                }}>
                  Chọn Camera Network <span style={{ color: 'var(--accent-secondary)' }}>*</span>
                </label>
                <select
                  className="input"
                  value={selectedNetworkId}
                  onChange={(e) => {
                    const val = e.target.value;
                    setSelectedNetworkId(val ? Number(val) : '');
                    setValidationError(null);
                  }}
                  disabled={loadingNetworks}
                  style={{
                    cursor: 'pointer',
                    appearance: 'auto',
                    padding: '10px 12px',
                    fontSize: '0.9rem'
                  }}
                >
                  <option value="">-- Chọn Camera Network --</option>
                  {networks.map((net) => (
                    <option key={net.id} value={net.id}>
                      {net.name} {net.status === 'running' ? '• (Đang hoạt động)' : ''}
                    </option>
                  ))}
                </select>
                {loadingNetworks && (
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '4px', display: 'block' }}>
                    Đang tải danh sách mạng camera...
                  </span>
                )}
              </div>

              {/* 2. Drag & Drop target upload area */}
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  color: 'white',
                  marginBottom: '8px'
                }}>
                  Ảnh mục tiêu tìm kiếm <span style={{ color: 'var(--accent-secondary)' }}>*</span>
                </label>

                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  style={{
                    border: dragOver
                      ? '2px dashed var(--accent-primary)'
                      : preview
                        ? '1px solid var(--border-color)'
                        : '2px dashed var(--border-color)',
                    borderRadius: 'var(--radius-lg)',
                    padding: preview ? '12px' : '32px 16px',
                    textAlign: 'center',
                    cursor: 'pointer',
                    position: 'relative',
                    backgroundColor: dragOver
                      ? 'rgba(99, 102, 241, 0.08)'
                      : preview
                        ? 'var(--bg-primary)'
                        : 'rgba(39, 43, 54, 0.2)',
                    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: '200px',
                    overflow: 'hidden'
                  }}
                >
                  {preview ? (
                    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                      <img
                        src={preview}
                        alt="Target preview"
                        style={{
                          maxWidth: '100%',
                          maxHeight: '220px',
                          borderRadius: 'var(--radius-md)',
                          display: 'block',
                          margin: '0 auto',
                          boxShadow: '0 8px 16px rgba(0,0,0,0.3)'
                        }}
                      />

                      {/* Hover info or clear button */}
                      <button
                        onClick={handleClearImage}
                        style={{
                          position: 'absolute',
                          top: '6px',
                          right: '6px',
                          width: '28px',
                          height: '28px',
                          borderRadius: '50%',
                          backgroundColor: 'rgba(15, 17, 21, 0.8)',
                          border: '1px solid rgba(255,255,255,0.1)',
                          color: 'white',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          cursor: 'pointer',
                          transition: 'background-color 0.2s'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--error)'}
                        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'rgba(15, 17, 21, 0.8)'}
                      >
                        <X size={16} />
                      </button>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                      <div style={{
                        width: '48px',
                        height: '48px',
                        borderRadius: '12px',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        border: '1px solid rgba(99, 102, 241, 0.2)',
                        color: 'var(--accent-primary)',
                        marginBottom: '4px'
                      }}>
                        <Upload size={22} />
                      </div>
                      <span style={{ fontWeight: 600, color: 'white', fontSize: '0.9rem' }}>
                        Kéo & thả ảnh ở đây
                      </span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', padding: '0 12px' }}>
                        Định dạng hỗ trợ JPG, PNG. Nhấp chuột để mở bộ chọn file.
                      </span>
                    </div>
                  )}
                  <input
                    type="file"
                    onChange={handleFileChange}
                    accept="image/*"
                    style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer' }}
                  />
                </div>
              </div>

              {/* Validation Warning Alert */}
              {validationError && (
                <div style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '8px',
                  backgroundColor: 'rgba(239, 68, 68, 0.1)',
                  border: '1px solid rgba(239, 68, 68, 0.2)',
                  borderRadius: 'var(--radius-md)',
                  padding: '10px 12px',
                  fontSize: '0.8rem',
                  color: '#f87171'
                }}>
                  <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
                  <span>{validationError}</span>
                </div>
              )}

              {/* Action search button */}
              <Button
                onClick={handleSearch}
                disabled={loading || !file || !selectedNetworkId}
                style={{
                  width: '100%',
                  padding: '12px',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '0.95rem',
                  fontWeight: 600,
                  marginTop: '4px'
                }}
              >
                {loading ? (
                  <>
                    <RefreshCw size={18} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
                    Đang truy vết Re-ID...
                  </>
                ) : (
                  <>
                    <SearchIcon size={18} />
                    Truy vết đối tượng
                  </>
                )}
              </Button>
            </div>
          </Card>
        </div>

        {/* Results Box */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>

          {/* Header section with counts */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            borderBottom: '1px solid var(--border-color)',
            paddingBottom: '12px'
          }}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
              Kết quả tìm kiếm
              {hasSearched && !loading && (
                <span style={{
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-secondary)',
                  padding: '2px 8px',
                  borderRadius: '10px',
                  border: '1px solid var(--border-color)'
                }}>
                  Tìm thấy {results.length} đối tượng
                </span>
              )}
            </h3>
          </div>

          {/* Results body */}
          {loading ? (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '80px 20px',
              gap: '16px'
            }}>
              <RefreshCw size={48} color="var(--accent-primary)" style={{ animation: 'spin 1.5s linear infinite' }} />
              <div style={{ textAlign: 'center' }}>
                <span style={{ display: 'block', fontWeight: 600, color: 'white', fontSize: '1.1rem' }}>
                  Đang xử lý ảnh đầu vào & so khớp đặc trưng...
                </span>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px', display: 'block' }}>
                  Hệ thống đang trích xuất đặc trưng Re-ID (512-dim) và tính toán khoảng cách Cosine trên cơ sở dữ liệu.
                </span>
              </div>
            </div>
          ) : results.length > 0 ? (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(310px, 1fr))',
              gap: 'var(--spacing-lg)'
            }}>
              {results.map((group, i) => (
                <MatchItem key={i} group={group} />
              ))}
            </div>
          ) : hasSearched ? (
            <div style={{
              backgroundColor: 'rgba(239, 68, 68, 0.04)',
              border: '1px dashed rgba(239, 68, 68, 0.25)',
              borderRadius: 'var(--radius-lg)',
              padding: '48px 24px',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '12px'
            }}>
              <AlertTriangle size={36} color="var(--error)" />
              <div>
                <span style={{ display: 'block', fontWeight: 600, color: 'white', fontSize: '1.05rem' }}>
                  Không tìm thấy đối tượng trùng khớp
                </span>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '6px', display: 'block', maxWidth: '500px', margin: '6px auto 0' }}>
                  Không tìm thấy đặc trưng khớp vượt ngưỡng cosine hoặc không có track dữ liệu nào cho Camera Network này. Vui lòng đảm bảo camera network đã được chạy tracking để lưu vết trước khi tìm kiếm.
                </span>
              </div>
            </div>
          ) : (
            <div style={{
              border: '1px dashed var(--border-color)',
              borderRadius: 'var(--radius-lg)',
              padding: '64px 24px',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '16px',
              backgroundColor: 'rgba(30, 32, 40, 0.1)'
            }}>
              <div style={{
                width: '56px',
                height: '56px',
                borderRadius: '50%',
                backgroundColor: 'var(--bg-tertiary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--text-secondary)',
                border: '1px solid var(--border-color)'
              }}>
                <FileImage size={24} />
              </div>
              <div style={{ maxWidth: '440px' }}>
                <span style={{ display: 'block', fontWeight: 600, color: 'white', fontSize: '1rem' }}>
                  Chờ tải ảnh mục tiêu & Cấu hình
                </span>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '6px', display: 'block' }}>
                  Hãy chọn Camera Network có sẵn, tải lên hình ảnh chân dung hoặc toàn thân của đối tượng cần truy vết để bắt đầu tìm kiếm thông tin hành trình.
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Global CSS for spinner animation */}
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        .animate-spin {
          animation: spin 1s linear infinite;
        }
      `}</style>
    </div>
  );
}
