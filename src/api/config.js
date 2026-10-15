// Central API configuration.
// Set VITE_API_BASE_URL and VITE_USE_MOCK=false in a .env file to hit the real backend.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';
export const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false';

// Simulated network latency for the mock layer (ms).
export const MOCK_LATENCY = 350;
