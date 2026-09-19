/**
 * Centralized API Client with JWT Bearer Interceptors.
 */

import axios from 'axios';
import {
  AuthResponse,
  BenchmarkResponse,
  KnowledgeComponent,
  QuestionPublic,
  SessionDetailedAnalytics,
  SubmitAnswerResult,
  TutorDecision,
  User,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor: attach Bearer token from localStorage
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('rl_tutor_token');
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Auth Endpoints
export const authApi = {
  register: (data: { username: string; email: string; password: string; role?: string }) =>
    apiClient.post<User>('/auth/register', data).then((res) => res.data),
  login: (data: { username: string; password: string }) =>
    apiClient.post<AuthResponse>('/auth/login', data).then((res) => res.data),
  getCurrentUser: () => apiClient.get<User>('/auth/me').then((res) => res.data),
};

// Curriculum Endpoints
export const curriculumApi = {
  getKCs: () => apiClient.get<KnowledgeComponent[]>('/curriculum/kcs').then((res) => res.data),
  getQuestions: (kcIdx?: number, difficulty?: string) =>
    apiClient
      .get('/curriculum/questions', { params: { kc_idx: kcIdx, difficulty } })
      .then((res) => res.data),
};

// Tutoring Session Endpoints
export const tutorApi = {
  startSession: (policyType: string = 'd3qn') =>
    apiClient
      .post<{ session_id: number; policy_type: string; initial_beliefs: number[]; mean_belief: number }>(
        '/tutor/sessions/start',
        { policy_type: policyType }
      )
      .then((res) => res.data),
  getNextAction: (sessionId: number) =>
    apiClient.get<TutorDecision>(`/tutor/sessions/${sessionId}/next-action`).then((res) => res.data),
  submitAnswer: (
    sessionId: number,
    data: {
      question_id: string;
      student_answer: number;
      response_time_seconds: number;
      hint_viewed: boolean;
      worked_example_viewed: boolean;
    }
  ) =>
    apiClient
      .post<SubmitAnswerResult>(`/tutor/sessions/${sessionId}/submit-answer`, data)
      .then((res) => res.data),
};

// Analytics & Benchmarking Endpoints
export const analyticsApi = {
  listSessions: () => apiClient.get<any[]>('/analytics/sessions').then((res) => res.data),
  getSessionAnalytics: (sessionId: number) =>
    apiClient.get<SessionDetailedAnalytics>(`/analytics/sessions/${sessionId}`).then((res) => res.data),
  getDashboardStats: () => apiClient.get<any>('/analytics/dashboard/stats').then((res) => res.data),
  runBenchmark: (episodes: number = 15, maxSteps: number = 25) =>
    apiClient
      .post<BenchmarkResponse>('/benchmarks/run', { episodes, max_steps: maxSteps })
      .then((res) => res.data),
};
