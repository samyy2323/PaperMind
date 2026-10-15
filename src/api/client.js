import axios from 'axios';
import { API_BASE_URL } from './config';

export const http = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

http.interceptors.request.use((config) => {
  try {
    const raw = localStorage.getItem('papermind.auth');
    const token = raw ? JSON.parse(raw).token : null;
    if (token) config.headers.Authorization = `Bearer ${token}`;
  } catch {
    /* ignore */
  }
  return config;
});

export const data = (promise) => promise.then((res) => res.data);
