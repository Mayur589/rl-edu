import React from 'react';
import { ArrowRight, CheckCircle, ChevronRight, Sparkles, TrendingUp, XCircle, Zap } from 'lucide-react';
import { useTutorStore } from '../../store/useTutorStore';

export const FeedbackModal: React.FC = () => {
  const {
    feedbackModalOpen,
    setFeedbackModalOpen,
    lastResult,
    fetchNextAction,
    isDecisionLoading,
    endSession,
  } = useTutorStore();

  if (!feedbackModalOpen || !lastResult) return null;

  const {
    is_correct,
    student_answer,
    correct_answer,
    explanation,
    new_mean_belief,
    learning_gain,
    reward_breakdown,
    is_session_completed,
  } = lastResult;

  const handleNext = async () => {
    if (is_session_completed) {
      endSession();
    } else {
      await fetchNextAction();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/85 backdrop-blur-md animate-in fade-in">
      <div className="w-full max-w-lg bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-2xl overflow-hidden">
        {/* Result Header */}
        <div className="flex items-center gap-3 pb-6 border-b border-slate-800">
          {is_correct ? (
            <div className="h-12 w-12 rounded-2xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center shrink-0">
              <CheckCircle className="h-7 w-7 text-emerald-400" />
            </div>
          ) : (
            <div className="h-12 w-12 rounded-2xl bg-rose-500/20 border border-rose-500/40 flex items-center justify-center shrink-0">
              <XCircle className="h-7 w-7 text-rose-400" />
            </div>
          )}
          <div>
            <h3 className="text-xl font-bold text-white">
              {is_correct ? 'Correct! Excellent Work.' : 'Incorrect Answer'}
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Your answer: <span className="font-mono text-slate-200">{student_answer}</span>
              {!is_correct && (
                <>
                  {' '}• Correct answer: <span className="font-mono text-emerald-400">{correct_answer}</span>
                </>
              )}
            </p>
          </div>
        </div>

        {/* Explanation */}
        <div className="py-5 space-y-3">
          <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300 leading-relaxed">
            <span className="font-semibold text-slate-100 block mb-1">Pedagogical Solution:</span>
            {explanation}
          </div>

          {/* Belief & Reward Telemetry Cards */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-xl bg-indigo-950/20 border border-indigo-500/20">
              <div className="flex items-center gap-1 text-[11px] font-semibold text-indigo-300 mb-1">
                <TrendingUp className="h-3.5 w-3.5" />
                <span>Mean BKT Mastery</span>
              </div>
              <div className="text-lg font-bold text-white">
                {(new_mean_belief * 100).toFixed(1)}%
              </div>
              <span className={`text-[10px] font-medium ${learning_gain >= 0 ? 'text-emerald-400' : 'text-slate-500'}`}>
                {learning_gain >= 0 ? `+${(learning_gain * 100).toFixed(2)}% gain` : 'no change'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-violet-950/20 border border-violet-500/20">
              <div className="flex items-center gap-1 text-[11px] font-semibold text-violet-300 mb-1">
                <Zap className="h-3.5 w-3.5" />
                <span>RL Reward Signal</span>
              </div>
              <div className="text-lg font-bold text-white">
                {reward_breakdown.total_reward >= 0 ? `+${reward_breakdown.total_reward}` : reward_breakdown.total_reward}
              </div>
              <span className="text-[10px] text-slate-400">
                NLG: {reward_breakdown.nlg_reward} • Step: {reward_breakdown.step_penalty}
              </span>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="pt-4 border-t border-slate-800 flex items-center justify-between">
          <span className="text-[11px] text-slate-500">
            {is_session_completed ? 'Curriculum target reached!' : 'D3QN policy optimizing next item'}
          </span>
          <button
            onClick={handleNext}
            disabled={isDecisionLoading}
            className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs flex items-center gap-1.5 transition-all shadow-lg shadow-indigo-600/20 disabled:opacity-50"
          >
            <span>{is_session_completed ? 'Finish Session' : 'Next Action'}</span>
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
