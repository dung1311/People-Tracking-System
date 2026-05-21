import { useState, useEffect } from 'react';
import { Card } from '../components/Common/Card';
import { Button } from '../components/Common/Button';
import { ConfigService } from '../api/services';
import type { TrackingConfig } from '../types';
import { Settings, Plus, FileCode, CheckCircle, Trash2, Edit2 } from 'lucide-react';

export function Configs() {
  const [configs, setConfigs] = useState<TrackingConfig[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<TrackingConfig | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  
  // Form fields
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [configType, setConfigType] = useState<'sct' | 'mct'>('sct');
  const [configDataStr, setConfigDataStr] = useState('{}');
  const [isDefault, setIsDefault] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const userRole = localStorage.getItem('user_role');
  const canEdit = userRole === 'ADMIN' || userRole === 'OPERATOR';

  const loadConfigs = async () => {
    try {
      const data = await ConfigService.getAll();
      setConfigs(data);
      if (data.length > 0 && !selectedConfig) {
        setSelectedConfig(data[0]);
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    loadConfigs();
  }, []);

  useEffect(() => {
    if (selectedConfig) {
      setName(selectedConfig.name);
      setDescription(selectedConfig.description || '');
      setConfigType(selectedConfig.config_type);
      setConfigDataStr(JSON.stringify(selectedConfig.config_data, null, 2));
      setIsDefault(selectedConfig.is_default);
    }
  }, [selectedConfig]);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    let parsedConfig = {};
    try {
      parsedConfig = JSON.parse(configDataStr);
    } catch (err) {
      setError('Cấu hình JSON không hợp lệ. Vui lòng kiểm tra lại cú pháp.');
      return;
    }

    const payload = {
      name,
      description,
      config_type: configType,
      config_data: parsedConfig,
      is_default: isDefault
    };

    try {
      if (isCreating) {
        const newCfg = await ConfigService.create(payload);
        setSuccess('Tạo cấu hình mới thành công!');
        setIsCreating(false);
        setIsEditing(false);
        setSelectedConfig(newCfg);
      } else if (selectedConfig && selectedConfig.id) {
        const updated = await ConfigService.update(selectedConfig.id, payload);
        setSuccess('Cập nhật cấu hình thành công!');
        setIsEditing(false);
        setSelectedConfig(updated);
      }
      loadConfigs();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Lưu cấu hình thất bại');
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm('Bạn có chắc chắn muốn xóa cấu hình này?')) return;
    setError(null);
    setSuccess(null);

    try {
      await ConfigService.delete(id);
      setSuccess('Xóa cấu hình thành công!');
      setSelectedConfig(null);
      loadConfigs();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Xóa cấu hình thất bại');
    }
  };

  const handleStartCreate = () => {
    setIsCreating(true);
    setIsEditing(true);
    setSelectedConfig(null);
    setName('');
    setDescription('');
    setConfigType('sct');
    setIsDefault(false);
    setConfigDataStr(JSON.stringify({
      DETECTOR: { conf_thres: 0.25, iou_thres: 0.45 },
      TRACKER: { max_age: 30, n_init: 3 }
    }, null, 2));
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-lg)' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.5px' }}>Quản lý Cấu hình</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Tùy chỉnh thông số nhận dạng Single-Cam (SCT) và Multi-Cam (MCT)</p>
        </div>
        {canEdit && (
          <Button onClick={handleStartCreate} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Plus size={18} />
            Tạo cấu hình mới
          </Button>
        )}
      </div>

      {error && (
        <div style={{ padding: '0.75rem 1rem', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', color: 'var(--error)', fontSize: '0.9rem' }}>
          {error}
        </div>
      )}
      {success && (
        <div style={{ padding: '0.75rem 1rem', backgroundColor: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)', borderRadius: '8px', color: 'var(--success)', fontSize: '0.9rem' }}>
          {success}
        </div>
      )}

      {/* Main Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: 'var(--spacing-lg)' }}>
        {/* Left: List of configs */}
        <Card style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-secondary)', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>Danh sách cấu hình</h3>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '60vh', overflowY: 'auto' }}>
            {configs.map((cfg) => (
              <div 
                key={cfg.id}
                onClick={() => { setSelectedConfig(cfg); setIsEditing(false); setIsCreating(false); }}
                style={{
                  padding: '10px 12px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  backgroundColor: selectedConfig?.id === cfg.id ? 'var(--bg-tertiary)' : 'transparent',
                  border: selectedConfig?.id === cfg.id ? '1px solid var(--accent-primary)' : '1px solid transparent',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  transition: 'all 0.2s ease'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                  <FileCode size={18} color={cfg.config_type === 'sct' ? 'var(--accent-primary)' : 'var(--accent-secondary)'} />
                  <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    <span style={{ fontWeight: 500, fontSize: '0.9rem', display: 'block' }}>{cfg.name}</span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                      {cfg.config_type} {cfg.is_default && '• Mặc định'}
                    </span>
                  </div>
                </div>
                {cfg.is_default && <CheckCircle size={16} color="var(--success)" />}
              </div>
            ))}
          </div>
        </Card>

        {/* Right: Selected config detail or editor */}
        <Card style={{ padding: 'var(--spacing-xl)' }}>
          {selectedConfig || isCreating ? (
            <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ fontSize: '1.25rem', fontWeight: 600 }}>
                  {isCreating ? 'Tạo cấu hình mới' : isEditing ? 'Chỉnh sửa cấu hình' : 'Thông tin cấu hình'}
                </h3>
                
                <div style={{ display: 'flex', gap: '8px' }}>
                  {!isEditing && canEdit && selectedConfig && (
                    <>
                      <Button type="button" onClick={() => setIsEditing(true)} style={{ display: 'flex', alignItems: 'center', gap: '6px', backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', color: 'white' }}>
                        <Edit2 size={16} />
                        Sửa
                      </Button>
                      {!selectedConfig.is_default && selectedConfig.id && (
                        <Button type="button" onClick={() => handleDelete(selectedConfig.id!)} style={{ display: 'flex', alignItems: 'center', gap: '6px', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', color: 'var(--error)' }}>
                          <Trash2 size={16} />
                          Xóa
                        </Button>
                      )}
                    </>
                  )}
                  {isEditing && (
                    <>
                      <Button type="submit" style={{ backgroundColor: 'var(--success)', border: 'none' }}>Lưu thay đổi</Button>
                      <Button type="button" onClick={() => { setIsEditing(false); setIsCreating(false); loadConfigs(); }} style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', color: 'white' }}>Hủy</Button>
                    </>
                  )}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>Tên cấu hình</label>
                  <input
                    type="text"
                    disabled={!isEditing}
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      backgroundColor: 'var(--bg-secondary)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '6px',
                      color: 'white',
                      fontSize: '0.9rem'
                    }}
                    required
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>Loại cấu hình</label>
                  <select
                    disabled={!isEditing || !isCreating}
                    value={configType}
                    onChange={(e) => {
                      const type = e.target.value as 'sct' | 'mct';
                      setConfigType(type);
                      if (isCreating) {
                        if (type === 'mct') {
                          setConfigDataStr(JSON.stringify({
                            MATCHING: { thresholds: { spatial_thresh: 50.0, reid_thresh: 0.45, time_thresh: 30 } },
                            GLOBAL_TRACK: { max_lost_frames: 100, confirm_frames: 3 }
                          }, null, 2));
                        } else {
                          setConfigDataStr(JSON.stringify({
                            DETECTOR: { conf_thres: 0.25, iou_thres: 0.45 },
                            TRACKER: { max_age: 30, n_init: 3 }
                          }, null, 2));
                        }
                      }
                    }}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      backgroundColor: 'var(--bg-secondary)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '6px',
                      color: 'white',
                      fontSize: '0.9rem'
                    }}
                  >
                    <option value="sct">Single-Cam Tracking (SCT)</option>
                    <option value="mct">Multi-Cam Tracking (MCT)</option>
                  </select>
                </div>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>Mô tả chi tiết</label>
                <textarea
                  disabled={!isEditing}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  style={{
                    width: '100%',
                    height: '60px',
                    padding: '8px 12px',
                    backgroundColor: 'var(--bg-secondary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    color: 'white',
                    fontSize: '0.9rem',
                    resize: 'none'
                  }}
                />
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="checkbox"
                  id="is_default"
                  disabled={!isEditing}
                  checked={isDefault}
                  onChange={(e) => setIsDefault(e.target.checked)}
                  style={{ width: '16px', height: '16px', cursor: 'pointer' }}
                />
                <label htmlFor="is_default" style={{ fontSize: '0.9rem', fontWeight: 500, cursor: 'pointer' }}>
                  Đặt làm cấu hình mặc định hệ thống cho loại này
                </label>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>Thông số cấu hình (JSON Format)</label>
                <textarea
                  disabled={!isEditing}
                  value={configDataStr}
                  onChange={(e) => setConfigDataStr(e.target.value)}
                  style={{
                    width: '100%',
                    height: '240px',
                    padding: '12px',
                    backgroundColor: '#12141a',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    color: '#a5d6ff',
                    fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    lineHeight: '1.4'
                  }}
                  required
                />
              </div>
            </form>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '300px', color: 'var(--text-secondary)' }}>
              <Settings size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
              <p>Chọn một cấu hình để xem chi tiết hoặc tạo cấu hình mới</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
