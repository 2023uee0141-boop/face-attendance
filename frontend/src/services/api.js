/**
 * API Service
 * 
 * Centralized API client using Axios.
 * All backend communication goes through this service.
 */

import axios from 'axios';

// Create axios instance with base configuration
const API = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5001/api',
  timeout: 60000, // 60s timeout (AI processing can take time)
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - attach JWT token
API.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor - handle auth errors
API.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      // Redirect to login if not already there
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// ==================== AUTH API ====================

export const authAPI = {
  /**
   * Register a new admin or teacher
   */
  register: (data) => API.post('/auth/register', data),

  /**
   * Login with email and password
   */
  login: (data) => API.post('/auth/login', data),

  /**
   * Get current authenticated user
   */
  getMe: () => API.get('/auth/me'),
};

// ==================== STUDENT API ====================

export const studentAPI = {
  /**
   * Register a new student with face image
   * @param {Object} data - { name, rollNumber, image (base64) }
   */
  register: (data) => API.post('/students/register', data),

  /**
   * Get all registered students
   */
  getAll: () => API.get('/students'),

  /**
   * Get a student by ID
   */
  getById: (id) => API.get(`/students/${id}`),

  /**
   * Delete a student
   */
  delete: (id) => API.delete(`/students/${id}`),
};

// ==================== ATTENDANCE API ====================

export const attendanceAPI = {
  /**
   * Mark attendance by sending face image
   * @param {Object} data - { image (base64) }
   */
  mark: (data) => API.post('/attendance/mark', data),

  /**
   * Get all attendance records with optional filters
   * @param {Object} params - { page, limit, date, studentId }
   */
  getAll: (params) => API.get('/attendance', { params }),

  /**
   * Get today's attendance
   */
  getToday: () => API.get('/attendance/today'),

  /**
   * Get attendance for a specific student
   */
  getByStudent: (studentId) => API.get(`/attendance/student/${studentId}`),
};

export default API;
