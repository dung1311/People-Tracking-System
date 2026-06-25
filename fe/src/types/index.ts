export interface Camera {
  id?: number;
  network_id?: number;
  name: string;
  source: string;
  source_type: string;
  description?: string;
  is_active: boolean;
  is_primary: boolean;
  config_path?: string;
  location?: string;
  resolution?: string;
  fps?: number;
  has_calibration: boolean;
  calibration_path?: string;
  created_at?: string;
  updated_at?: string;
}

export interface CameraCreate {
  network_id?: number;
  name: string;
  source: string;
  source_type?: string;
  description?: string;
  is_active?: boolean;
  is_primary?: boolean;
  location?: string;
}

export interface User {
  id: number;
  username: string;
  email: string;
  role: 'ADMIN' | 'OPERATOR' | 'VIEWER';
  is_active: boolean;
  created_at: string;
}

export interface TrackingConfig {
  id?: number;
  name: string;
  description?: string;
  config_type: 'sct' | 'mct';
  config_data: Record<string, any>;
  is_default: boolean;
  created_at?: string;
  updated_at?: string;
}


export interface CameraNetwork {
  cameras?: Camera[];
  id?: number;
  name: string;
  status: 'created' | 'running' | 'stopping' | 'completed' | 'failed';
  sct_config: Record<string, any>;
  mct_config: Record<string, any>;
  started_at?: string;
  stopped_at?: string;
  output_video_path?: string;
  output_txt_dir?: string;
  total_frames: number;
  total_global_ids: number;
  avg_fps: number;
  created_at?: string;
}


export interface SearchMatch {
  camera_id: number;
  camera_name?: string;
  person_id: number;
  best_score: number;
  start_time: string;
  end_time: string;
  count: number;
  distance: number;
  best_match: {
    id: number;
    bbox: number[];
    score: number;
    timestamp: string;
    frame_id: number;
    distance: number;
  };
}

export interface SearchResult {
  matches: SearchMatch[];
}

export interface Track {
  id: number;
  camera_id: number;
  person_id: number;
  frame_id: number;
  bbox: number[];
  score: number;
  class_id: number;
  timestamp: string;
}