/**
 * Zustand State Store for RL Tutor Platform.
 */

import { create } from 'zustand';
import { analyticsApi, authApi, tutorApi } from '../api/client';
import {
  BenchmarkResponse,
  SessionDetailedAnalytics,
  SubmitAnswerResult,
  TutorDecision,
  User,
} from '../types';

interface TutorState {
  // Auth State
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isAuthLoading: boolean;

  // Tutoring Session State
  sessionId: number | null;
  policyType: string;
  isSessionActive: boolean;
  currentStep: number;
  currentDecision: TutorDecision | null;
  lastResult: SubmitAnswerResult | null;
  liveBeliefs: number[];
  meanBelief: number;
  reviewUrgency: number;
  isSubmitting: boolean;
  isDecisionLoading: boolean;

  // Modals & Scaffolding UI
  hintDrawerOpen: boolean;
  workedExampleModalOpen: boolean;
  feedbackModalOpen: boolean;

  // Analytics & Research State
  selectedSessionAnalytics: SessionDetailedAnalytics | null;
  benchmarkResults: BenchmarkResponse | null;
  isBenchmarking: boolean;
  error: string | null;

  // Actions
  initAuth: () => Promise<void>;
  login: (credentials: { username: string; password: string }) => Promise<void>;
  register: (data: { username: string; email: string; password: string; role?: string }) => Promise<void>;
  logout: () => void;
  clearError: () => void;

  startSession: (policyType?: string) => Promise<void>;
  fetchNextAction: () => Promise<void>;
  submitAnswer: (
    answer: number,
    responseTimeSeconds: number,
    hintViewed: boolean,
    workedExampleViewed: boolean
  ) => Promise<void>;
  endSession: () => void;

  setHintDrawerOpen: (open: boolean) => void;
  setWorkedExampleModalOpen: (open: boolean) => void;
  setFeedbackModalOpen: (open: boolean) => void;

  fetchSessionAnalytics: (sessionId: number) => Promise<void>;
  runBenchmark: (episodes?: number, maxSteps?: number) => Promise<void>;
}

export const useTutorStore = create<TutorState>((set, get) => ({
  user: null,
  token: localStorage.getItem('rl_tutor_token'),
  isAuthenticated: !!localStorage.getItem('rl_tutor_token'),
  isAuthLoading: false,

  sessionId: null,
  policyType: 'd3qn',
  isSessionActive: false,
  currentStep: 0,
  currentDecision: null,
  lastResult: null,
  liveBeliefs: [0.2, 0.2, 0.2, 0.2],
  meanBelief: 0.2,
  reviewUrgency: 0.0,
  isSubmitting: false,
  isDecisionLoading: false,

  hintDrawerOpen: false,
  workedExampleModalOpen: false,
  feedbackModalOpen: false,

  selectedSessionAnalytics: null,
  benchmarkResults: null,
  isBenchmarking: false,
  error: null,

  initAuth: async () => {
    const token = localStorage.getItem('rl_tutor_token');
    if (!token) return;
    set({ isAuthLoading: true });
    try {
      const user = await authApi.getCurrentUser();
      set({ user, isAuthenticated: true, isAuthLoading: false });
    } catch {
      localStorage.removeItem('rl_tutor_token');
      set({ user: null, token: null, isAuthenticated: false, isAuthLoading: false });
    }
  },

  login: async (credentials) => {
    set({ isAuthLoading: true, error: null });
    try {
      const res = await authApi.login(credentials);
      localStorage.setItem('rl_tutor_token', res.access_token);
      set({ token: res.access_token, isAuthenticated: true, isAuthLoading: false });
      const user = await authApi.getCurrentUser();
      set({ user });
    } catch (err: any) {
      set({
        error: err.response?.data?.detail || 'Invalid username or password',
        isAuthLoading: false,
      });
      throw err;
    }
  },

  register: async (data) => {
    set({ isAuthLoading: true, error: null });
    try {
      await authApi.register(data);
      // Auto login after registration
      await get().login({ username: data.username, password: data.password });
    } catch (err: any) {
      set({
        error: err.response?.data?.detail || 'Registration failed',
        isAuthLoading: false,
      });
      throw err;
    }
  },

  logout: () => {
    localStorage.removeItem('rl_tutor_token');
    set({
      user: null,
      token: null,
      isAuthenticated: false,
      sessionId: null,
      isSessionActive: false,
      currentDecision: null,
      lastResult: null,
    });
  },

  clearError: () => set({ error: null }),

  startSession: async (policyType = 'd3qn') => {
    set({ isDecisionLoading: true, error: null });
    try {
      const res = await tutorApi.startSession(policyType);
      set({
        sessionId: res.session_id,
        policyType: res.policy_type,
        isSessionActive: true,
        currentStep: 0,
        liveBeliefs: res.initial_beliefs,
        meanBelief: res.mean_belief,
        lastResult: null,
      });
      await get().fetchNextAction();
    } catch (err: any) {
      set({
        error: err.response?.data?.detail || 'Failed to initialize tutoring session',
        isDecisionLoading: false,
      });
    }
  },

  fetchNextAction: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    set({ isDecisionLoading: true, error: null, feedbackModalOpen: false });
    try {
      const decision = await tutorApi.getNextAction(sessionId);
      set({
        currentDecision: decision,
        currentStep: decision.step_number,
        liveBeliefs: decision.beliefs,
        meanBelief: decision.mean_belief,
        reviewUrgency: decision.review_urgency,
        isDecisionLoading: false,
      });
    } catch (err: any) {
      set({
        error: err.response?.data?.detail || 'Failed to fetch next instructional action',
        isDecisionLoading: false,
      });
    }
  },

  submitAnswer: async (answer, responseTime, hintViewed, workedExampleViewed) => {
    const { sessionId, currentDecision } = get();
    if (!sessionId || !currentDecision) return;

    set({ isSubmitting: true, error: null });
    try {
      const result = await tutorApi.submitAnswer(sessionId, {
        question_id: currentDecision.question.id,
        student_answer: answer,
        response_time_seconds: responseTime,
        hint_viewed: hintViewed,
        worked_example_viewed: workedExampleViewed,
      });

      set({
        lastResult: result,
        liveBeliefs: result.new_beliefs,
        meanBelief: result.new_mean_belief,
        feedbackModalOpen: true,
        isSubmitting: false,
        isSessionActive: !result.is_session_completed,
      });
    } catch (err: any) {
      set({
        error: err.response?.data?.detail || 'Failed to evaluate student submission',
        isSubmitting: false,
      });
    }
  },

  endSession: () => {
    set({
      sessionId: null,
      isSessionActive: false,
      currentDecision: null,
      lastResult: null,
      feedbackModalOpen: false,
      hintDrawerOpen: false,
      workedExampleModalOpen: false,
    });
  },

  setHintDrawerOpen: (open) => set({ hintDrawerOpen: open }),
  setWorkedExampleModalOpen: (open) => set({ workedExampleModalOpen: open }),
  setFeedbackModalOpen: (open) => set({ feedbackModalOpen: open }),

  fetchSessionAnalytics: async (sessionId: number) => {
    try {
      const data = await analyticsApi.getSessionAnalytics(sessionId);
      set({ selectedSessionAnalytics: data });
    } catch (err: any) {
      set({ error: err.response?.data?.detail || 'Failed to load session analytics' });
    }
  },

  runBenchmark: async (episodes = 15, maxSteps = 25) => {
    set({ isBenchmarking: true, error: null });
    try {
      const results = await analyticsApi.runBenchmark(episodes, maxSteps);
      set({ benchmarkResults: results, isBenchmarking: false });
    } catch (err: any) {
      set({
        error: err.response?.data?.detail || 'Benchmark execution failed',
        isBenchmarking: false,
      });
    }
  },
}));
