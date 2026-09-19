import React from 'react';
import { Award, CheckCircle, Clock } from 'lucide-react';
import { useTutorStore } from '../../store/useTutorStore';

const KC_LABELS = [
  'Basic Arithmetic',
  'Advanced Arithmetic',
  'Basic Algebra',
  'Advanced Algebra',
];

export const MasteryProgressBar: React.FC = () => {
  const { liveBeliefs, meanBelief, reviewUrgency } = useTutorStore();

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Award className="h-4 w-4 text-indigo-400" />
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
            Cognitive Mastery Beliefs (BKT)
          </h4>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-400">Mean Mastery:</span>
          <span className="font-mono font-bold text-indigo-400">
            {(meanBelief * 100).toFixed(1)}%
          </span>
          {reviewUrgency > 0.4 && (
            <span className="px-2 py-0.5 rounded-md text-[10px] font-semibold bg-amber-500/10 text-amber-300 border border-amber-500/30">
              Decay Detected
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {KC_LABELS.map((name, idx) => {
          const belief = liveBeliefs[idx] || 0.2;
          const pct = Math.round(belief * 100);
          const isMastered = belief >= 0.88;

          return (
            <div key={name} className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80">
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="text-slate-300 font-medium truncate">{name}</span>
                <span className="font-mono font-bold text-slate-200">{pct}%</span>
              </div>
              <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden relative">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isMastered
                      ? 'bg-gradient-to-r from-emerald-500 to-emerald-400 shadow-sm shadow-emerald-500/50'
                      : pct > 50
                      ? 'bg-gradient-to-r from-indigo-500 to-indigo-400'
                      : 'bg-slate-600'
                  }`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="flex items-center justify-between mt-2 text-[10px] text-slate-500 font-mono">
                <span>{isMastered ? 'Mastered' : pct > 50 ? 'Proficient' : 'Developing'}</span>
                {isMastered && <CheckCircle className="h-3 w-3 text-emerald-400 inline" />}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
