export interface Camera {
  id: number;
  name: string;
  source: string;
  description?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface CameraCreate {
  name: string;
  source: string;
  description?: string;
  is_active?: boolean;
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

export interface SearchResult {
  matches: Track[];
}