import { useState, useEffect } from 'react';
import { CameraService, CameraRoiService } from '../api/services';
import { Video, Users, Clock, Maximize } from 'lucide-react';

export function RoiAnalysis() {
  const [cameras, setCameras] = useState<any[]>([]);
  
  // Segments and Analysis state
  const [segments, setSegments] = useState<any[]>([]);
  const [selectedSegmentId, setSelectedSegmentId] = useState<number | null>(null);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [analyzing, setAnalyzing] = useState(false);

  // Fetch initial data
  const loadInitialData = async () => {
    try {
      const cams = await CameraService.getAll();
      setCameras(cams);
      
      const segs = await CameraRoiService.getAllSegments();
      setSegments(segs);
    } catch (err) {
      console.error('Error fetching initial data:', err);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  const handleAnalyze = async () => {
    if (selectedSegmentId === null) {
      alert('Vui lòng chọn một file output / phân đoạn video.');
      return;
    }
    
    setAnalyzing(true);
    setAnalysisResults(null);
    try {
      const results = await CameraRoiService.analyzeRoi(selectedSegmentId);
      setAnalysisResults(results);
    } catch (err) {
      console.error(err);
      alert('Có lỗi xảy ra khi phân tích.');
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      {/* Title */}
      <div>
        <h2 style={{ fontSize: '1.8rem', fontWeight: 800, color: 'white', margin: 0 }}>Phân tích ROI</h2>
        <p style={{ color: 'var(--text-secondary)', margin: '8px 0 0 0' }}>
          Xem khu vực giám sát (ROI) và thống kê mật độ người, thời gian tập trung sau khi chạy xong pipeline.
        </p>
      </div>


      {/* Analysis Section */}
      <div style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)',
        borderRadius: '16px',
        padding: '24px',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
        display: 'flex',
        flexDirection: 'column',
        gap: '24px'
      }}>
        <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '16px' }}>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'white', margin: 0 }}>Phân tích sau khi chạy (Post-run ROI Analysis)</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: '4px 0 0 0' }}>
            Chọn video output đã chạy xong pipeline để phân tích thời gian tập trung và số lượng người.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-end' }}>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Chọn phân đoạn video (Output)</label>
            <select
              value={selectedSegmentId || ''}
              onChange={(e) => setSelectedSegmentId(Number(e.target.value))}
              style={{
                backgroundColor: 'var(--bg-tertiary)',
                color: 'white',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                padding: '10px 14px',
                fontSize: '0.9rem'
              }}
            >
              <option value="">-- Chọn một Video Output --</option>
              {segments.map(s => {
                const camera = cameras.find(c => c.id === s.camera_id);
                const cameraName = camera ? camera.name : `Camera ${s.camera_id}`;
                const fps = camera ? (camera.fps || 25) : 25;
                const frameCount = Math.round((s.duration_seconds || 0) * fps);
                const timeStr = new Date(s.start_time).toLocaleString();
                return (
                  <option key={s.id} value={s.id}>
                    {cameraName} | {timeStr}, {frameCount} frame
                  </option>
                );
              })}
            </select>
            {segments.length === 0 && (
              <p style={{ color: 'var(--accent-secondary)', fontSize: '0.8rem', marginTop: '6px', margin: 0 }}>
                * Chưa có video segment nào hoàn thành. Vui lòng chạy pipeline và bấm Stop hoặc chờ phân đoạn kết thúc để lưu dữ liệu vào hệ thống.
              </p>
            )}
          </div>
          <button
            onClick={handleAnalyze}
            disabled={analyzing}
            className="btn btn-primary"
            style={{ height: '42px', padding: '0 24px', display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            {analyzing ? (
              <span className="spin-animation" style={{
                width: '16px',
                height: '16px',
                border: '2px solid white',
                borderTopColor: 'transparent',
                borderRadius: '50%',
                display: 'inline-block'
              }} />
            ) : (
              <Video size={16} />
            )}
            Bắt đầu phân tích
          </button>
        </div>

        {/* Analysis Results Display */}
        {analysisResults && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', marginTop: '16px' }}>
            {Object.values(analysisResults).map((res: any) => (
              <div key={res.roi_id} style={{
                border: '1px solid var(--border-color)',
                borderRadius: '12px',
                padding: '24px',
                backgroundColor: 'var(--bg-tertiary)'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', marginBottom: '20px' }}>
                  <h4 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'white', margin: 0 }}>
                    Khu vực: <span style={{ color: 'var(--accent-primary)' }}>{res.roi_name}</span>
                  </h4>
                </div>

                {/* Key stats row */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: '16px',
                  marginBottom: '32px'
                }}>
                  <div style={{ backgroundColor: 'var(--bg-secondary)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <div style={{ color: 'var(--accent-secondary)' }}><Users size={24} /></div>
                    <div>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block' }}>Tổng số lượt người</span>
                      <strong style={{ fontSize: '1.4rem', color: 'white' }}>{res.total_people}</strong>
                    </div>
                  </div>
                  <div style={{ backgroundColor: 'var(--bg-secondary)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <div style={{ color: '#f59e0b' }}><Clock size={24} /></div>
                    <div>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block' }}>TG tập trung TB</span>
                      <strong style={{ fontSize: '1.4rem', color: 'white' }}>{res.average_dwell_time_seconds}s</strong>
                    </div>
                  </div>
                  <div style={{ backgroundColor: 'var(--bg-secondary)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <div style={{ color: '#34d399' }}><Maximize size={24} /></div>
                    <div>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block' }}>Mật độ đỉnh (cùng lúc)</span>
                      <strong style={{ fontSize: '1.4rem', color: 'white' }}>{res.max_occupancy} người</strong>
                    </div>
                  </div>
                </div>

                {/* Occupancy over time & People list */}
                <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
                  {/* Occupancy Chart */}
                  <div>
                    <h5 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'white', marginBottom: '16px' }}>Mật độ theo thời gian</h5>
                    {res.occupancy_over_time.length === 0 ? (
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Không có dữ liệu mật độ.</p>
                    ) : (
                      <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '8px',
                        backgroundColor: 'var(--bg-secondary)',
                        padding: '16px',
                        borderRadius: '8px',
                        border: '1px solid var(--border-color)',
                        height: '240px',
                        overflowY: 'auto'
                      }}>
                        {res.occupancy_over_time.map((occ: any, index: number) => (
                          <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', width: '70px' }}>
                              {new Date(occ.timestamp).toLocaleTimeString()}
                            </span>
                            <div style={{ flex: 1, height: '14px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', overflow: 'hidden' }}>
                              <div style={{
                                width: `${Math.min(100, (occ.count / (res.max_occupancy || 1)) * 100)}%`,
                                height: '100%',
                                backgroundColor: 'var(--accent-secondary)'
                              }} />
                            </div>
                            <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'white', width: '20px', textAlign: 'right' }}>
                              {occ.count}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* People list */}
                  <div>
                    <h5 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'white', marginBottom: '16px' }}>Danh sách chi tiết đối tượng</h5>
                    <div style={{
                      backgroundColor: 'var(--bg-secondary)',
                      borderRadius: '8px',
                      border: '1px solid var(--border-color)',
                      maxHeight: '240px',
                      overflowY: 'auto'
                    }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                        <thead>
                          <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-secondary)', textAlign: 'left' }}>
                            <th style={{ padding: '10px 16px' }}>ID đối tượng</th>
                            <th style={{ padding: '10px 16px' }}>TG trong vùng</th>
                            <th style={{ padding: '10px 16px' }}>Vào lúc</th>
                          </tr>
                        </thead>
                        <tbody>
                          {res.people_metrics.map((p: any) => (
                            <tr key={p.person_id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                              <td style={{ padding: '10px 16px', color: 'white', fontWeight: 600 }}>#{p.person_id}</td>
                              <td style={{ padding: '10px 16px', color: 'var(--accent-secondary)' }}>{p.dwell_time_seconds}s</td>
                              <td style={{ padding: '10px 16px', color: 'var(--text-secondary)' }}>{new Date(p.entered_at).toLocaleTimeString()}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
