import { useState, useEffect } from 'react';
import { SystemService } from '../api/services';
import { Cpu, HardDrive, HelpCircle, RefreshCw } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  progress?: number;
  icon: React.ReactNode;
  color: string;
}

function MetricCard({ title, value, subtitle, progress, icon, color }: MetricCardProps) {
  return (
    <div style={{
      backgroundColor: 'var(--bg-secondary)',
      border: '1px solid var(--border-color)',
      borderRadius: '16px',
      padding: '24px',
      display: 'flex',
      flexDirection: 'column',
      gap: '16px',
      position: 'relative',
      overflow: 'hidden',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
      transition: 'transform 0.2s ease, box-shadow 0.2s ease',
    }}
    onMouseEnter={(e) => {
      e.currentTarget.style.transform = 'translateY(-2px)';
      e.currentTarget.style.boxShadow = `0 8px 30px ${color}15`;
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.transform = 'translateY(0)';
      e.currentTarget.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.15)';
    }}>
      {/* Decorative gradient overlay */}
      <div style={{
        position: 'absolute',
        top: 0,
        right: 0,
        width: '120px',
        height: '120px',
        background: `radial-gradient(circle, ${color}10 0%, transparent 70%)`,
        pointerEvents: 'none',
      }} />

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-secondary)' }}>{title}</span>
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '10px',
          backgroundColor: `${color}15`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: color,
        }}>
          {icon}
        </div>
      </div>

      <div>
        <h3 style={{ fontSize: '2rem', fontWeight: 800, color: 'white', margin: 0 }}>{value}</h3>
        {subtitle && <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '4px 0 0 0' }}>{subtitle}</p>}
      </div>

      {progress !== undefined && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '4px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Sử dụng</span>
            <span style={{ fontWeight: 600, color: 'white' }}>{progress.toFixed(1)}%</span>
          </div>
          <div style={{ width: '100%', height: '6px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '3px', overflow: 'hidden' }}>
            <div style={{
              width: `${progress}%`,
              height: '100%',
              backgroundColor: color,
              borderRadius: '3px',
              transition: 'width 0.5s ease-out'
            }} />
          </div>
        </div>
      )}
    </div>
  );
}

export function SystemStatus() {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const refreshInterval = 3000; // 3 seconds

  const fetchMetrics = async () => {
    try {
      const data = await SystemService.getStatus();
      setMetrics(data);
      setError(null);
    } catch (err: any) {
      console.error(err);
      setError('Không thể kết nối đến máy chủ hoặc lấy thông tin hệ thống.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '1.8rem', fontWeight: 800, color: 'white', margin: 0 }}>Trạng thái hệ thống</h2>
          <p style={{ color: 'var(--text-secondary)', margin: '8px 0 0 0' }}>
            Giám sát tài nguyên máy chủ theo thời gian thực (CPU, RAM, GPU).
          </p>
        </div>
        <button
          onClick={() => {
            setLoading(true);
            fetchMetrics();
          }}
          className="btn btn-secondary"
          style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <RefreshCw size={16} className={loading ? 'spin-animation' : ''} />
          Làm mới
        </button>
      </div>

      {error && (
        <div style={{
          padding: '16px',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid var(--error)',
          borderRadius: '12px',
          color: 'var(--error)',
          fontSize: '0.9rem',
        }}>
          {error}
        </div>
      )}

      {loading && !metrics ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '100px 0' }}>
          <div className="spin-animation" style={{
            width: '40px',
            height: '40px',
            border: '4px solid var(--border-color)',
            borderTopColor: 'var(--accent-primary)',
            borderRadius: '50%',
          }} />
        </div>
      ) : (
        metrics && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
            {/* Cards Grid */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
              gap: '24px',
            }}>
              {/* CPU Card */}
              <MetricCard
                title="CPU STATUS"
                value={`${metrics.cpu?.usage_percent?.toFixed(1) || 0}%`}
                subtitle={`${metrics.cpu?.cores || 0} nhân logic`}
                progress={metrics.cpu?.usage_percent || 0}
                icon={<Cpu size={20} />}
                color="var(--accent-primary)"
              />

              {/* RAM Card */}
              <MetricCard
                title="RAM STATUS"
                value={`${metrics.ram?.used_gb || 0} GB / ${metrics.ram?.total_gb || 0} GB`}
                subtitle={`Còn trống ${metrics.ram?.free_gb || 0} GB`}
                progress={metrics.ram?.usage_percent || 0}
                icon={<HardDrive size={20} />}
                color="var(--accent-secondary)"
              />
            </div>

            {/* GPU Section */}
            <div>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'white', marginBottom: '16px' }}>
                Xử lý đồ họa (GPU)
              </h3>
              {metrics.gpu && metrics.gpu.length > 0 ? (
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
                  gap: '24px',
                }}>
                  {metrics.gpu.map((gpu: any, idx: number) => (
                    <div
                      key={idx}
                      style={{
                        backgroundColor: 'var(--bg-secondary)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '16px',
                        padding: '24px',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '20px',
                        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div>
                          <h4 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'white', margin: 0 }}>
                            {gpu.name}
                          </h4>
                          <span style={{ fontSize: '0.75rem', color: 'var(--accent-secondary)', fontWeight: 600, textTransform: 'uppercase', marginTop: '4px', display: 'inline-block' }}>
                            NVIDIA DEVICE #{idx}
                          </span>
                        </div>
                      </div>

                      {/* GPU Core Load */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>GPU Core Load</span>
                          <span style={{ fontWeight: 600, color: 'white' }}>{gpu.utilization_percent}%</span>
                        </div>
                        <div style={{ width: '100%', height: '6px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{
                            width: `${gpu.utilization_percent}%`,
                            height: '100%',
                            backgroundColor: '#34d399',
                            borderRadius: '3px',
                          }} />
                        </div>
                      </div>

                      {/* GPU Memory usage */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Bộ nhớ đồ họa (VRAM)</span>
                          <span style={{ fontWeight: 600, color: 'white' }}>
                            {gpu.memory_used_mb} MB / {gpu.memory_total_mb} MB
                          </span>
                        </div>
                        <div style={{ width: '100%', height: '6px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{
                            width: `${(gpu.memory_used_mb / gpu.memory_total_mb) * 100}%`,
                            height: '100%',
                            backgroundColor: '#f59e0b',
                            borderRadius: '3px',
                          }} />
                        </div>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', alignSelf: 'flex-end' }}>
                          Trống {gpu.memory_free_mb} MB
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{
                  padding: '32px',
                  backgroundColor: 'var(--bg-secondary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '16px',
                  textAlign: 'center',
                  color: 'var(--text-secondary)',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '12px',
                }}>
                  <HelpCircle size={32} color="var(--text-secondary)" />
                  <div>
                    <span style={{ fontWeight: 600, color: 'white', display: 'block' }}>Không tìm thấy GPU tương thích</span>
                    <span style={{ fontSize: '0.85rem' }}>Hệ thống không phát hiện driver NVIDIA hoặc card đồ họa rời.</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )
      )}
    </div>
  );
}
