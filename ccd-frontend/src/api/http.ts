import axios from 'axios';

export const http = axios.create({
  baseURL: '/api',
  // Long-running inference can exceed 60s; 0 disables Axios timeout
  timeout: 0,
});

http.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err?.response?.data?.detail ?? err.message ?? 'Request failed';
    return Promise.reject(new Error(msg));
  },
);

export type Http = typeof http;


