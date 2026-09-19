import React, { useState } from 'react';
import { AlertCircle, BrainCircuit, CheckCircle2, Lock, Mail, Sparkles, User as UserIcon } from 'lucide-react';
import { useTutorStore } from '../../store/useTutorStore';

export const AuthModal: React.FC = () => {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<'student' | 'researcher'>('student');

  const { login, register, isAuthLoading, error, clearError } = useTutorStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();
    if (isRegister) {
      await register({ username, email, password, role });
    } else {
      await login({ username, password });
    }
  };

  const handleQuickDemo = async (demoRole: 'student' | 'researcher') => {
    clearError();
    const demoUser = demoRole === 'student' ? 'demo_student' : 'demo_researcher';
    const demoEmail = `${demoUser}@rltutor.edu`;
    const demoPass = 'password123';

    try {
      await login({ username: demoUser, password: demoPass });
    } catch {
      // If not yet created, register then login
      await register({ username: demoUser, email: demoEmail, password: demoPass, role: demoRole });
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md">
      <div className="w-full max-w-md bg-slate-900/95 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-2xl shadow-indigo-500/10">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="inline-flex h-12 w-12 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 p-0.5 mb-3 shadow-lg shadow-indigo-500/30">
            <div className="h-full w-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <BrainCircuit className="h-6 w-6 text-indigo-400" />
            </div>
          </div>
          <h2 className="text-2xl font-bold text-white tracking-tight">
            {isRegister ? 'Create Researcher/Student Account' : 'Welcome to RL Tutor'}
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Explainable Reinforcement Learning Adaptive Learning Platform
          </p>
        </div>

        {/* Quick Demo Access */}
        <div className="mb-6 p-3 rounded-xl bg-indigo-950/40 border border-indigo-800/40">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-indigo-300 mb-2">
            <Sparkles className="h-3.5 w-3.5" />
            <span>Instant Demo Access (One-Click)</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => handleQuickDemo('student')}
              disabled={isAuthLoading}
              className="px-3 py-2 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 border border-indigo-500/30 text-indigo-200 text-xs font-medium transition-colors text-center"
            >
              Demo Student
            </button>
            <button
              type="button"
              onClick={() => handleQuickDemo('researcher')}
              disabled={isAuthLoading}
              className="px-3 py-2 rounded-lg bg-violet-600/20 hover:bg-violet-600/30 border border-violet-500/30 text-violet-200 text-xs font-medium transition-colors text-center"
            >
              Demo Researcher
            </button>
          </div>
        </div>

        <div className="relative mb-6">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-slate-800" />
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-slate-900 px-2 text-slate-500">or continue with credentials</span>
          </div>
        </div>

        {/* Error notification */}
        {error && (
          <div className="mb-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center gap-2 text-xs text-rose-300">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">Username</label>
            <div className="relative">
              <UserIcon className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
                className="w-full pl-9 pr-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 transition-colors"
              />
            </div>
          </div>

          {isRegister && (
            <>
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5">Email</label>
                <div className="relative">
                  <Mail className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="student@example.edu"
                    className="w-full pl-9 pr-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 transition-colors"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5">Role</label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setRole('student')}
                    className={`py-2 px-3 rounded-xl text-xs font-medium border transition-all ${
                      role === 'student'
                        ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300'
                        : 'bg-slate-950 border-slate-800 text-slate-400'
                    }`}
                  >
                    Student
                  </button>
                  <button
                    type="button"
                    onClick={() => setRole('researcher')}
                    className={`py-2 px-3 rounded-xl text-xs font-medium border transition-all ${
                      role === 'researcher'
                        ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300'
                        : 'bg-slate-950 border-slate-800 text-slate-400'
                    }`}
                  >
                    Researcher
                  </button>
                </div>
              </div>
            </>
          )}

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">Password</label>
            <div className="relative">
              <Lock className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-9 pr-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 transition-colors"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isAuthLoading}
            className="w-full py-2.5 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm transition-colors shadow-lg shadow-indigo-600/20 disabled:opacity-50"
          >
            {isAuthLoading ? 'Please wait...' : isRegister ? 'Create Account' : 'Sign In'}
          </button>
        </form>

        {/* Toggle between register & login */}
        <div className="mt-6 text-center">
          <button
            type="button"
            onClick={() => {
              setIsRegister(!isRegister);
              clearError();
            }}
            className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            {isRegister
              ? 'Already have an account? Sign in'
              : "Don't have an account? Create one"}
          </button>
        </div>
      </div>
    </div>
  );
};
