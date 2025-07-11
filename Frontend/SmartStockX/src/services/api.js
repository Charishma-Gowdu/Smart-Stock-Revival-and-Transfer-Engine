import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

const api = {
  get: async (endpoint) => {
    const response = await axiosInstance.get(endpoint);
    return response.data;
  },

  post: async (endpoint, data, isFormData = false) => {
    const headers = isFormData
      ? { 'Content-Type': 'multipart/form-data' }
      : { 'Content-Type': 'application/json' };

    const response = await axiosInstance.post(endpoint, data, { headers });
    return response.data;
  },
};

export default api;
