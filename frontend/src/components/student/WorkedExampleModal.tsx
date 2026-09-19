import React from 'react';
import { BookOpen, CheckCircle2, X } from 'lucide-react';
import { useTutorStore } from '../../store/useTutorStore';

export const WorkedExampleModal: React.FC = () => {
  const { currentDecision, workedExampleModalOpen, setWorkedExampleModalOpen } = useTutorStore();

  if (!workedExampleModalOpen || !currentDecision?.question) return null;

  const { worked_example_prompt, worked_example_explanation } = currentDecision.question;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-in fade-in">
      <div className="w-full max-w-xl bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-2xl">
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-2 text-indigo-400">
            <BookOpen className="h-5 w-5" />
            <h3 className="font-bold text-lg text-white">Demonstration Worked Example</h3>
          </div>
          <button
            onClick={() => setWorkedExampleModalOpen(false)}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="py-6 space-y-4">
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
            <span className="text-[11px] font-mono uppercase text-indigo-400 font-semibold tracking-wider">
              Sample Problem Demonstration
            </span>
            <p className="text-base text-slate-200 mt-1 font-medium">
              {worked_example_prompt}
            </p>
          </div>

          <div className="p-4 rounded-xl bg-indigo-950/20 border border-indigo-500/20">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-indigo-300 mb-1">
              <CheckCircle2 className="h-4 w-4 text-indigo-400" />
              <span>Step-by-Step Walkthrough</span>
            </div>
            <p className="text-sm text-indigo-100/90 leading-relaxed font-mono">
              {worked_example_explanation}
            </p>
          </div>
        </div>

        <div className="pt-4 border-t border-slate-800 flex justify-end">
          <button
            onClick={() => setWorkedExampleModalOpen(false)}
            className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-colors"
          >
            Understood, Return to Problem
          </button>
        </div>
      </div>
    </div>
  );
};
