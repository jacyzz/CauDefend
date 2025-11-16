import axios from 'axios';

export const http = axios.create({
  baseURL: '/api',
  timeout: 60_000,
});

http.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err?.response?.data?.detail ?? err.message ?? 'Request failed';
    return Promise.reject(new Error(msg));
  },
);

export type Http = typeof http;


