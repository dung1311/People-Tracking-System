import { useEffect, useState } from 'react';
import { Card } from '../components/Common/Card';
import { CameraStreamGrid } from '../components/CameraStreamGrid';
import { CameraService } from '../api/services';
import { Camera, Activity } from 'lucide-react';
import type { Camera as CameraType } from '../types';

export function Dashboard() {
  const [stats, setStats] = useState({ cameras: 0, tracks: 0 });
  const [cameras, setCameras] = useState<CameraType[]>([]);

  useEffect(() => {
    // Fetch generic stats (mock or real)
    CameraService.getAll().then(data => {
        setCameras(data);
        setStats(s => ({ ...s, cameras: data.length }));
    });
    // For tracks we might just show a static number or fetch recent count if api supports
  }, []);

  return (
    <div className="animate-fade-in">
      <h2 style={{ fontSize: '1.875rem', fontWeight: 700, marginBottom: 'var(--spacing-xl)' }}>Dashboard</h2>
      
      <div className="grid-cols-3">
        <Card>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                <div style={{ padding: '12px', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '12px' }}>
                    <Camera size={24} color="var(--accent-primary)" />
                </div>
                <div>
                    <div className="text-secondary text-sm">Active Cameras</div>
                    <div className="text-lg" style={{ fontSize: '2rem' }}>{stats.cameras}</div>
                </div>
            </div>
        </Card>

        <Card>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                <div style={{ padding: '12px', background: 'rgba(236, 72, 153, 0.1)', borderRadius: '12px' }}>
                    <Activity size={24} color="var(--accent-secondary)" />
                </div>
                <div>
                    <div className="text-secondary text-sm">System Status</div>
                    <div className="text-lg" style={{ fontSize: '1.5rem', color: 'var(--success)' }}>Operational</div>
                </div>
            </div>
        </Card>
      </div>

      <div style={{ marginTop: 'var(--spacing-xl)' }}>
        <h3 style={{ marginBottom: '1rem', fontSize: '1.25rem' }}>Live Views</h3>
        <CameraStreamGrid cameras={cameras} />
      </div>

    </div>
  );
}
