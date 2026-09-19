import React, { useState } from 'react';
import { ArrowRight, Bot, Compass, Cpu, Flame, Layers, Play, Sparkles } from 'lucide-react';
import { FeedbackModal } from '../components/student/FeedbackModal';
import { HintDrawer } from '../components/student/HintDrawer';
import { MasteryProgressBar } from '../components/student/MasteryProgressBar';
import { QuestionCard } from '../components/student/QuestionCard';
import { WorkedExampleModal } from '../components/student/WorkedExampleModal';
import { useTutorStore } from '../store/useTutorStore';

export const StudentTutorPage: React.FC = () => {
  const { isSessionActive, startSession, policyType, isDecisionLoading, endSession } = useTutorStore();
  const [selectedPolicy, setSelectedPolicy] = useState<string>('d3qn');

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Session Header / Setup */}
      {!isSessionActive ? (
        <div className="max-w-2xl mx-auto bg-slate-900/80 border border-slate-800 rounded-3xl p-8 sm:p-10 shadow-2xl text-center space-y-6">
          <div className="inline-flex h-16 w-16 rounded-2xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-violet-400 p-0.5 shadow-xl shadow-indigo-500/20">
            <div className="h-full w-full bg-slate-950 rounded-[14px] flex items-center justify-center">
              <Bot className="h-8 w-8 text-indigo-400" />
            </div>
          </div>

          <div>
            <h2 className="text-3xl font-extrabold text-white tracking-tight">
              Adaptive Intelligent Tutoring
            </h2>
            <p className="text-sm text-slate-400 mt-2 max-w-md mx-auto leading-relaxed">
              Experience reinforcement learning that personalizes difficulty, delivers hints,
              prevents cognitive friction, and schedules spaced review for long-term retention.
            </p>
          </div>

          {/* Policy Selector */}
          <div className="text-left space-y-2">
            <label className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Select Tutoring Engine Policy
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {[
                {
                  id: 'd3qn',
                  title: 'D3QN (Explainable RL)',
                  desc: 'Dueling Double DQN with Content & Guidance adaptation',
                  tag: 'Recommended',
                },
                {
                  id: 'heuristic',
                  title: 'Expert Rule-Based',
                  desc: 'Heuristic thresholds on rolling friction and accuracy',
                  tag: 'Baseline',
                },
                {
                  id: 'linear',
                  title: 'Static Linear Curriculum',
                  desc: 'Fixed sequential progression without guidance',
                  tag: 'Control',
                },
                {
                  id: 'random',
                  title: 'Uniform Random',
                  desc: 'Unbiased random action exploration',
                  tag: 'Ablation',
                },
              ].map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => setSelectedPolicy(p.id)}
                  className={`p-3.5 rounded-xl border text-left transition-all ${
                    selectedPolicy === p.id
                      ? 'bg-indigo-950/50 border-indigo-500 ring-1 ring-indigo-500 shadow-md shadow-indigo-950'
                      : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-200">{p.title}</span>
                    <span
                      className={`text-[9px] font-semibold uppercase px-1.5 py-0.5 rounded ${
                        p.id === 'd3qn'
                          ? 'bg-indigo-500/20 text-indigo-300'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {p.tag}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 leading-normal">{p.desc}</p>
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={() => startSession(selectedPolicy)}
            disabled={isDecisionLoading}
            className="w-full py-3.5 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-bold text-sm flex items-center justify-center gap-2 shadow-xl shadow-indigo-600/30 transition-all disabled:opacity-50"
          >
            <Play className="h-4 w-4 fill-white" />
            <span>{isDecisionLoading ? 'Initializing Session...' : 'Launch Learning Session'}</span>
          </button>
        </div>
      ) : (
        /* Active Tutoring Interface */
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />
              <h2 className="text-lg font-bold text-white tracking-tight">Active Learning Session</h2>
              <span className="text-xs text-slate-500 font-mono">
                (Policy: {policyType.toUpperCase()})
              </span>
            </div>

            <button
              onClick={endSession}
              className="text-xs text-slate-400 hover:text-rose-400 transition-colors px-3 py-1.5 rounded-lg hover:bg-slate-800/60"
            >
              Finish Early
            </button>
          </div>

          <MasteryProgressBar />

          <QuestionCard />
          <HintDrawer />
          <WorkedExampleModal />
          <FeedbackModal />
        </div>
      )}
    </div>
  );
};
