import React, { useEffect, useRef, useState } from 'react';
import { AlertCircle, BookOpen, Clock, HelpCircle, Lightbulb, Play, Send, Zap } from 'lucide-react';
import { useTutorStore } from '../../store/useTutorStore';

export const QuestionCard: React.FC = () => {
  const {
    currentDecision,
    isDecisionLoading,
    isSubmitting,
    submitAnswer,
    setHintDrawerOpen,
    setWorkedExampleModalOpen,
    hintDrawerOpen,
  } = useTutorStore();

  const [studentAnswer, setStudentAnswer] = useState('');
  const [startTime, setStartTime] = useState<number>(Date.now());
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [hintViewed, setHintViewed] = useState<boolean>(false);
  const [workedExampleViewed, setWorkedExampleViewed] = useState<boolean>(false);

  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setStudentAnswer('');
    setStartTime(Date.now());
    setElapsedSeconds(0);
    setHintViewed(false);
    setWorkedExampleViewed(false);
    inputRef.current?.focus();
  }, [currentDecision?.step_number, currentDecision?.question.id]);

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [startTime]);

  if (isDecisionLoading || !currentDecision) {
    return (
      <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-12 text-center flex flex-col items-center justify-center animate-pulse">
        <div className="h-10 w-10 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mb-4" />
        <p className="text-sm text-slate-300 font-medium">
          RL Agent evaluating cognitive state & selecting next pedagogical action...
        </p>
        <p className="text-xs text-slate-500 mt-1">
          Computing Q(s, a) across Content, Guidance, and Review adaptations
        </p>
      </div>
    );
  }

  const { question, action_name, is_review_session, target_kc_name } = currentDecision;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!studentAnswer || isSubmitting) return;

    const numAnswer = parseFloat(studentAnswer);
    if (isNaN(numAnswer)) return;

    submitAnswer(numAnswer, elapsedSeconds, hintViewed, workedExampleViewed);
  };

  const getDifficultyBadge = (diff: string) => {
    switch (diff.toLowerCase()) {
      case 'easy':
        return 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30';
      case 'hard':
        return 'bg-rose-500/10 text-rose-300 border-rose-500/30';
      default:
        return 'bg-amber-500/10 text-amber-300 border-amber-500/30';
    }
  };

  return (
    <div className="bg-slate-900/70 border border-slate-800/90 rounded-2xl p-6 sm:p-8 shadow-xl shadow-slate-950/40 backdrop-blur-sm relative overflow-hidden">
      {/* Top Banner: Pedagogical Strategy Meta */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-6 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <span className="px-3 py-1 rounded-full text-xs font-semibold bg-indigo-950 text-indigo-300 border border-indigo-800/80 flex items-center gap-1.5">
            <Zap className="h-3 w-3 text-indigo-400" />
            <span>RL Action: {action_name}</span>
          </span>
          {is_review_session && (
            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-cyan-950 text-cyan-300 border border-cyan-800/80">
              Spaced Review Active
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 text-xs text-slate-400">
          <span className="px-2.5 py-1 rounded-lg bg-slate-800/60 font-medium">
            {target_kc_name}
          </span>
          <span className={`px-2.5 py-1 rounded-lg font-semibold border ${getDifficultyBadge(question.difficulty)}`}>
            {question.difficulty}
          </span>
          <div className="flex items-center gap-1 text-slate-400">
            <Clock className="h-3.5 w-3.5 text-slate-500" />
            <span>{elapsedSeconds}s</span>
          </div>
        </div>
      </div>

      {/* Main Question Display */}
      <div className="py-8">
        <span className="text-xs uppercase font-mono tracking-wider text-slate-500">
          Problem {currentDecision.step_number} • {question.id}
        </span>
        <h3 className="text-2xl sm:text-3xl font-bold text-white mt-2 mb-4 tracking-tight leading-snug">
          {question.prompt}
        </h3>
        <p className="text-xs text-slate-400">
          Enter your numeric answer below and press Submit (or hit Enter).
        </p>
      </div>

      {/* Answer Input and Controls */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <input
              ref={inputRef}
              type="number"
              step="any"
              required
              disabled={isSubmitting}
              value={studentAnswer}
              onChange={(e) => setStudentAnswer(e.target.value)}
              placeholder="Your answer (e.g. 12 or 4.5)"
              className="w-full px-4 py-3 bg-slate-950 border border-slate-700/80 rounded-xl text-lg font-medium text-white placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
            />
          </div>

          <button
            type="submit"
            disabled={!studentAnswer || isSubmitting}
            className="px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm flex items-center justify-center gap-2 transition-all shadow-lg shadow-indigo-600/30 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Send className="h-4 w-4" />
            <span>{isSubmitting ? 'Evaluating...' : 'Submit Answer'}</span>
          </button>
        </div>

        {/* Guidance Scaffolding Triggers */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-slate-800/60">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                setHintDrawerOpen(!hintDrawerOpen);
                setHintViewed(true);
              }}
              className="px-3 py-1.5 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/20 text-xs font-medium flex items-center gap-1.5 transition-colors"
            >
              <Lightbulb className="h-3.5 w-3.5" />
              <span>{hintDrawerOpen ? 'Hide Scaffolding Hint' : 'Request Hint'}</span>
            </button>

            <button
              type="button"
              onClick={() => {
                setWorkedExampleModalOpen(true);
                setWorkedExampleViewed(true);
              }}
              className="px-3 py-1.5 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 border border-indigo-500/20 text-xs font-medium flex items-center gap-1.5 transition-colors"
            >
              <BookOpen className="h-3.5 w-3.5" />
              <span>Study Worked Example</span>
            </button>
          </div>

          <p className="text-[11px] text-slate-500">
            Note: Asking for hints adjusts cognitive friction telemetry in the POMDP state.
          </p>
        </div>
      </form>
    </div>
  );
};
