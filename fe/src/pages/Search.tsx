import { useState, useRef } from 'react';
import { Card } from '../components/Common/Card';
import { Button } from '../components/Common/Button';
import { SearchService } from '../api/services';
import type { SearchResult } from '../api/services';
import { Upload, Search as SearchIcon, Clock } from 'lucide-react';

// Helper component to display match with bbox
function MatchItem({ group }: { group: SearchResult['matches'][0] }) {
    const imgRef = useRef<HTMLImageElement>(null);
    const [scale, setScale] = useState(1);
    
    const handleLoad = () => {
        if (imgRef.current) {
            const { naturalWidth, clientWidth } = imgRef.current;
            setScale(clientWidth / naturalWidth);
        }
    };

    const { best_match, count, start_time, end_time, camera_id } = group;
    const [x1, y1, x2, y2] = best_match.bbox;

    // Formatting time
    const formatTime = (iso: string) => new Date(iso).toLocaleTimeString();
    
    // Calculate similarity. Cosine distance is usually [0, 2]. 
    // 0 is identical. We assume we want 1 - distance.
    // If distance is small (e.g. 0.2), match is good (80%).
    const similarity = Math.max(0, (1 - best_match.distance) * 100);
    
    return (
        <Card>
            <div className="flex flex-col gap-2">
                <div style={{ position: 'relative', width: '100%', overflow: 'hidden', borderRadius: '8px', backgroundColor: 'black' }}>
                    <img 
                        ref={imgRef}
                        src={`http://localhost:8000/api/v1/frames/${camera_id}/${best_match.frame_id}`}
                        alt={`Person ${group.person_id}`}
                        style={{ width: '100%', display: 'block' }}
                        onLoad={handleLoad}
                    />
                    <div 
                        style={{
                            position: 'absolute',
                            left: x1 * scale,
                            top: y1 * scale,
                            width: (x2 - x1) * scale,
                            height: (y2 - y1) * scale,
                            border: '2px solid var(--success)',
                            boxShadow: '0 0 4px rgba(0,0,0,0.5)',
                            pointerEvents: 'none'
                        }}
                    >
                         <div style={{ 
                             position: 'absolute', 
                             top: '-20px', 
                             left: 0, 
                             background: 'var(--success)', 
                             color: 'black', 
                             fontSize: '10px', 
                             padding: '0 4px',
                             fontWeight: 'bold'
                         }}>
                             {similarity.toFixed(0)}%
                         </div>
                    </div>
                </div>
                
                <div className="flex justify-between items-start mt-2">
                    <div>
                        <div style={{ fontWeight: 600, fontSize: '1.1rem' }}>Person #{group.person_id}</div> 
                        <div className="text-secondary text-sm">Camera {camera_id}</div>
                    </div>
                    <div className="text-right">
                         <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.85rem' }} className="text-secondary">
                            <Clock size={14} />
                            {formatTime(start_time)} - {formatTime(end_time)}
                         </div>
                         <div style={{ fontSize: '0.8rem', color: 'var(--accent-primary)', fontWeight: 'bold' }}>
                            {count} sightings
                         </div>
                    </div>
                </div>
            </div>
        </Card>
    );
}

export function Search() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult['matches']>([]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const f = e.target.files[0];
      setFile(f);
      setPreview(URL.createObjectURL(f));
    }
  };

  const handleSearch = async () => {
    if (!file) return;
    setLoading(true);
    setResults([]);
    try {
      const data = await SearchService.searchPromise(file);
      setResults(data.matches);
    } catch (e) {
      alert('Search failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in">
      <h2 style={{ fontSize: '1.875rem', fontWeight: 700, marginBottom: 'var(--spacing-xl)' }}>Person Search</h2>

      <div className="grid-cols-2" style={{ alignItems: 'start' }}>
        <Card title="Upload Image">
            <div style={{ 
                border: '2px dashed var(--border-color)', 
                borderRadius: 'var(--radius-md)',
                padding: '2rem',
                textAlign: 'center',
                cursor: 'pointer',
                marginBottom: '1rem',
                position: 'relative'
            }}>
                {preview ? (
                    <img src={preview} style={{ maxHeight: '200px', margin: '0 auto', borderRadius: '8px' }} />
                ) : (
                    <div className="text-secondary flex flex-col items-center gap-2">
                        <Upload size={32} />
                        <span>Click to upload or drag image here</span>
                    </div>
                )}
                <input 
                    type="file" 
                    onChange={handleFileChange} 
                    accept="image/*"
                    style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer' }}
                />
            </div>
            <Button onClick={handleSearch} disabled={loading || !file} style={{ width: '100%' }}>
                {loading ? 'Searching...' : <><SearchIcon size={18} /> Search Matches</>}
            </Button>
        </Card>

        <div style={{ display: 'grid', gap: '1rem' }}>
            {results.map((group, i) => (
                <MatchItem key={i} group={group} />
            ))}
            {results.length === 0 && !loading && (
                <div className="text-secondary" style={{ textAlign: 'center', padding: '2rem' }}>
                    Results will appear here
                </div>
            )}
        </div>
      </div>
    </div>
  );
}
