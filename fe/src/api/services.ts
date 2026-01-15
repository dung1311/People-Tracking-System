import { apiClient } from './client';
import type { Camera, CameraCreate } from '../types';

export type { Camera };

// ... (keep Track and SearchResult interfaces)

export interface Track {
  id: number;
  camera_id: number;
  frame_id: number;
  bbox: [number, number, number, number];
  score: number;
  timestamp: string;
}

export interface GroupedMatch {
    camera_id: number;
    person_id: number;
    start_time: string;
    end_time: string;
    count: number;
    best_match: {
        id: number;
        bbox: [number, number, number, number];
        score: number;
        timestamp: string;
        frame_id: number;
    };
}

export interface SearchResult {
    matches: GroupedMatch[];
}

export const CameraService = {
  getAll: async () => {
    const response = await apiClient.get<Camera[]>('/cameras/');
    return response.data;
  },
  create: async (data: CameraCreate) => {
    const response = await apiClient.post<Camera>('/cameras/', data);
    return response.data;
  },
  delete: async (id: number) => {
    await apiClient.delete(`/cameras/${id}`);
  },
};

export const TrackService = {
    getAll: async (limit = 20) => {
      const response = await apiClient.get<Track[]>('/tracks/', { params: { limit }});
      return response.data;
    }
}

export const SearchService = {
  searchPromise: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await apiClient.post<SearchResult>('/search/search', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};
