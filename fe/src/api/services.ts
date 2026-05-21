import { apiClient } from './client';
import type { Camera, CameraCreate, User, TrackingConfig, CameraNetwork, Track, SearchResult } from '../types';

export type { Camera, Track, SearchResult, CameraNetwork };

// Authentication Service
export const AuthService = {
  login: async (username: string, password: string) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    const response = await apiClient.post<{ access_token: string; token_type: string }>('/auth/login', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  register: async (user: any) => {
    const response = await apiClient.post<User>('/auth/register', user);
    return response.data;
  },
  me: async () => {
    const response = await apiClient.get<User>('/auth/me');
    return response.data;
  },
};

// User Management Service (Admin only)
export const UserService = {
  getAll: async () => {
    const response = await apiClient.get<User[]>('/users/');
    return response.data;
  },
  update: async (id: number, data: any) => {
    const response = await apiClient.patch<User>(`/users/${id}`, data);
    return response.data;
  },
  delete: async (id: number) => {
    await apiClient.delete(`/users/${id}`);
  },
};

// Camera Management Service
export const CameraService = {
  getAll: async () => {
    const response = await apiClient.get<Camera[]>('/cameras/');
    return response.data;
  },
  create: async (data: CameraCreate) => {
    const response = await apiClient.post<Camera>('/cameras/', data);
    return response.data;
  },
  update: async (id: number, data: any) => {
    const response = await apiClient.patch<Camera>(`/cameras/${id}`, data);
    return response.data;
  },
  delete: async (id: number) => {
    await apiClient.delete(`/cameras/${id}`);
  },
  uploadCalibration: async (id: number, file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await apiClient.post<{ calibration_path: string }>(`/cameras/${id}/calibration`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  getCalibration: async (id: number) => {
    const response = await apiClient.get<Record<string, any>>(`/cameras/${id}/calibration`);
    return response.data;
  },
  deleteCalibration: async (id: number) => {
    await apiClient.delete(`/cameras/${id}/calibration`);
  },
  uploadVideo: async (id: number, file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await apiClient.post<{ source: string; resolution: string; fps: number }>(
      `/cameras/${id}/video`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  },
  getThumbnail: async (id: number) => {
    const response = await apiClient.get<{ url: string }>(`/cameras/${id}/thumbnail`);
    return response.data;
  },
  checkConnection: async (source: string, sourceType: string = 'rtsp') => {
    const response = await apiClient.post<{ ok: boolean; message: string; resolution?: string; fps?: number }>(
      '/cameras/check-connection',
      { source, source_type: sourceType }
    );
    return response.data;
  },
};

// Configuration Profile Service
export const ConfigService = {
  getAll: async (type?: 'sct' | 'mct') => {
    const response = await apiClient.get<TrackingConfig[]>('/configs/', {
      params: type ? { config_type: type } : {},
    });
    return response.data;
  },
  getDefaults: async () => {
    const response = await apiClient.get<TrackingConfig[]>('/configs/defaults');
    return response.data;
  },
  create: async (data: any) => {
    const response = await apiClient.post<TrackingConfig>('/configs/', data);
    return response.data;
  },
  update: async (id: number, data: any) => {
    const response = await apiClient.patch<TrackingConfig>(`/configs/${id}`, data);
    return response.data;
  },
  delete: async (id: number) => {
    await apiClient.delete(`/configs/${id}`);
  },
};

// Tracking Sessions Service
export const CameraNetworkService = {
  getAll: async () => {
    const response = await apiClient.get<CameraNetwork[]>('/camera_networks/');
    return response.data;
  },
  getById: async (id: number) => {
    const response = await apiClient.get<CameraNetwork>(`/camera_networks/${id}`);
    return response.data;
  },
  create: async (data: { name: string; camera_ids?: number[]; sct_config_id?: number; mct_config_id?: number }) => {
    const response = await apiClient.post<CameraNetwork>('/camera_networks/', data);
    return response.data;
  },
  delete: async (id: number) => {
    await apiClient.delete(`/camera_networks/${id}`);
  },
  start: async (id: number) => {
    const response = await apiClient.post<{ ok: boolean; status: string }>(`/camera_networks/${id}/start`);
    return response.data;
  },
  stop: async (id: number) => {
    const response = await apiClient.post<{ ok: boolean; status: string }>(`/camera_networks/${id}/stop`);
    return response.data;
  },
  getOutputUrl: async (id: number) => {
    const response = await apiClient.get<{ url: string }>(`/camera_networks/${id}/output`);
    return response.data;
  },
};

export const TrackService = {
  getAll: async (limit = 20) => {
    const response = await apiClient.get<Track[]>('/tracks/', { params: { limit } });
    return response.data;
  },
};

export const SearchService = {
  searchPromise: async (file: File, networkId?: number) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await apiClient.post<SearchResult>('/search/search', formData, {
      params: networkId ? { network_id: networkId } : {},
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};
