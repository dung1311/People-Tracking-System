import React, { useState } from 'react';
import type { Camera } from '../types'; 
import { Card } from './Common/Card';
import { Maximize2, X } from 'lucide-react';

interface Props {
    cameras: Camera[];
}

// Simple Modal Component
const Modal = ({ children, onClose }: { children: React.ReactNode, onClose: () => void }) => (
    <div style={{
        position: 'fixed',
        top: 0, left: 0, right: 0, bottom: 0,
        background: 'rgba(0,0,0,0.8)',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2rem'
    }}>
        <div style={{ position: 'relative', maxWidth: '90%', maxHeight: '90%' }}>
            <button 
                onClick={onClose}
                style={{
                    position: 'absolute',
                    top: -40, right: 0,
                    background: 'none', border: 'none',
                    color: 'white', cursor: 'pointer'
                }}
            >
                <X size={32} />
            </button>
            {children}
        </div>
    </div>
);

export const CameraStreamGrid: React.FC<Props> = ({ cameras }) => {
    const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null);

    const getStreamUrl = (id?: number) => id ? `${window.location.protocol}//${window.location.host}/api/v1/stream/camera/${id}` : '';

    return (
        <>
            <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', 
                gap: 'var(--spacing-lg)',
                marginTop: 'var(--spacing-xl)'
            }}>
                {cameras.map(cam => (
                    <Card key={cam.id} title={cam.name}>
                        <div 
                            style={{ 
                                position: 'relative', 
                                aspectRatio: '16/9', 
                                background: '#000', 
                                borderRadius: '8px', 
                                overflow: 'hidden',
                                cursor: 'pointer'
                            }}
                            className="group" 
                            onClick={() => setSelectedCamera(cam)}
                        >
                            {cam.id && (
                                <img 
                                    src={getStreamUrl(cam.id)} 
                                    alt={cam.name}
                                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                    onError={(e) => {
                                        (e.target as HTMLImageElement).style.display = 'none';
                                    }}
                                />
                            )}
                            {/* Overlay Icon */}
                            <div style={{
                                position: 'absolute',
                                top: '50%', left: '50%',
                                transform: 'translate(-50%, -50%)',
                                opacity: 0,
                                transition: 'opacity 0.2s',
                                background: 'rgba(0,0,0,0.5)',
                                padding: '1rem',
                                borderRadius: '50%'
                            }} 
                            onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                            onMouseLeave={(e) => e.currentTarget.style.opacity = '0'}
                            >
                                <Maximize2 color="white" />
                            </div>
                        </div>
                        <div style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                           {cam.is_active ? 'Live' : 'Offline'}
                        </div>
                    </Card>
                ))}
            </div>

            {selectedCamera && selectedCamera.id && (
                <Modal onClose={() => setSelectedCamera(null)}>
                    <div style={{ background: '#000', borderRadius: '8px', overflow: 'hidden' }}>
                         <img 
                            src={getStreamUrl(selectedCamera.id)} 
                            alt={selectedCamera.name}
                            style={{ maxWidth: '100%', maxHeight: '85vh', display: 'block' }}
                        />
                        <div style={{ padding: '1rem', color: 'white', background: '#222' }}>
                            <h3>{selectedCamera.name}</h3>
                        </div>
                    </div>
                </Modal>
            )}
        </>
    );
};
