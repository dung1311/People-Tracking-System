import { useEffect, useState } from 'react';
import { Card } from '../components/Common/Card';
import { Button } from '../components/Common/Button';
import { Input } from '../components/Common/Input';
import { CameraService } from '../api/services';
import type { Camera as ICamera } from '../api/services';
import { Plus, Trash2, Video } from 'lucide-react';

export function Cameras() {
  const [cameras, setCameras] = useState<ICamera[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newCamera, setNewCamera] = useState({ name: '', source: '' });

  useEffect(() => {
    loadCameras();
  }, []);

  const loadCameras = () => {
    CameraService.getAll().then(setCameras);
  };

  const handleAdd = async () => {
    if (!newCamera.name || !newCamera.source) return;
    await CameraService.create(newCamera);
    setIsModalOpen(false);
    setNewCamera({ name: '', source: '' });
    loadCameras();
  };

  const handleDelete = async (id: number) => {
    if(!window.confirm('Are you sure?')) return;
    await CameraService.delete(id);
    loadCameras();
  };

  return (
    <div className="animate-fade-in">
      <div className="flex justify-between items-center mb-4">
        <h2 style={{ fontSize: '1.875rem', fontWeight: 700 }}>Cameras</h2>
        <Button onClick={() => setIsModalOpen(true)}>
            <Plus size={18} /> Add Camera
        </Button>
      </div>

      {isModalOpen && (
        <div style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', 
            display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50
        }}>
            <Card className="animate-fade-in" style={{ width: '400px', padding: '2rem' }}>
                <h3 className="text-lg mb-4">Add New Camera</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <Input 
                        placeholder="Camera Name" 
                        value={newCamera.name}
                        onChange={e => setNewCamera({...newCamera, name: e.target.value})}
                    />
                    <Input 
                        placeholder="RTSP Source / URL" 
                        value={newCamera.source}
                        onChange={e => setNewCamera({...newCamera, source: e.target.value})}
                    />
                    <div className="flex gap-2 justify-between mt-4">
                        <Button variant="secondary" onClick={() => setIsModalOpen(false)} style={{width:'100%'}}>Cancel</Button>
                        <Button onClick={handleAdd} style={{width:'100%'}}>Add</Button>
                    </div>
                </div>
            </Card>
        </div>
      )}

      <div className="grid-cols-2">
        {cameras.map(cam => (
            <Card key={cam.id}>
                <div className="flex flex-col gap-4">
                    <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                            <div style={{ padding: '10px', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                                <Video size={20} color="var(--text-secondary)" />
                            </div>
                            <div>
                                <div style={{ fontWeight: 600 }}>{cam.name}</div>
                                <div className="text-secondary text-sm">{cam.source}</div>
                            </div>
                        </div>
                        <Button variant="icon" className="hover:text-error" onClick={() => cam.id && handleDelete(cam.id)}>
                            <Trash2 size={18} color="var(--error)" />
                        </Button>
                    </div>
                    
                    {/* Video Stream */}
                    <div className="aspect-video bg-black rounded-lg overflow-hidden relative group">
                        <img 
                            src={`http://localhost:8000/api/v1/cameras/${cam.id}/stream`} 
                            alt={`Stream from ${cam.name}`}
                            className="w-full h-full object-contain"
                            loading="lazy"
                            onError={(e) => {
                                // Fallback if stream fails
                                e.currentTarget.style.display = 'none';
                                e.currentTarget.parentElement?.classList.add('flex', 'items-center', 'justify-center');
                                const errDiv = document.createElement('div');
                                errDiv.textContent = 'Stream Offline';
                                errDiv.style.color = 'var(--text-secondary)';
                                e.currentTarget.parentElement?.appendChild(errDiv);
                            }}
                        />
                    </div>
                </div>
            </Card>
        ))}
      </div>
    </div>
  );
}
