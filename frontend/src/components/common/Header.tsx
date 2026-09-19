import React from 'react';
import { BrainCircuit, Compass, Flame, LogOut, ShieldCheck, User as UserIcon } from 'lucide-react';
import { useTutorStore } from '../../store/useTutorStore';

interface HeaderProps {
  activeTab: 'student' | 'researcher';
  setActiveTab: (tab: 'student' | 'researcher') => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab }) => {
  const { user, logout, isSessionActive, currentStep } = useTutorStore();

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800/80 bg-slate-950/85 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-violet-400 p-0.5 shadow-lg shadow-indigo-500/20">
            <div className="h-full w-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <BrainCircuit className="h-5 w-5 text-indigo-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg text-white tracking-tight">RL Tutor</span>
              <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-indigo-950 text-indigo-300 border border-indigo-800/60">
                POMDP • D3QN
              </span>
            </div>
            <p className="text-xs text-slate-400">Explainable Adaptive Learning Engine</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setActiveTab('student')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'student'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            <Compass className="h-3.5 w-3.5" />
            <span>Student Tutor</span>
            {isSessionActive && (
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            )}
          </button>
          <button
            onClick={() => setActiveTab('researcher')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'researcher'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            <ShieldCheck className="h-3.5 w-3.5" />
            <span>RL Dashboard</span>
          </button>
        </div>

        {/* User Info & Actions */}
        <div className="flex items-center gap-3">
          {user?.profile && (
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs font-semibold">
              <Flame className="h-3.5 w-3.5 text-amber-400" />
              <span>{user.profile.current_streak} streak</span>
            </div>
          )}

          <div className="flex items-center gap-2.5 pl-2 border-l border-slate-800">
            <div className="text-right hidden sm:block">
              <p className="text-xs font-semibold text-slate-200">{user?.username || 'Guest'}</p>
              <p className="text-[10px] text-slate-400 capitalize">{user?.role || 'Visitor'}</p>
            </div>
            <button
              onClick={logout}
              title="Sign Out"
              className="p-2 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
