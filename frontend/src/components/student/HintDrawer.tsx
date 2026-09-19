import React from 'react';
import { HelpCircle, Lightbulb, X } from 'lucide-react';
import { useTutorStore } from '../../store/useTutorStore';

export const HintDrawer: React.FC = () => {
  const { currentDecision, hintDrawerOpen, setHintDrawerOpen } = useTutorStore();

  if (!hintDrawerOpen || !currentDecision?.question) return null;

  return (
    <div className="mt-4 p-5 rounded-xl bg-amber-950/30 border border-amber-500/30 backdrop-blur-sm animate-in fade-in slide-in-from-top-2 duration-200">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 text-amber-300 text-sm font-semibold">
          <Lightbulb className="h-4 w-4 text-amber-400" />
          <span>Scaffolding Guidance Hint</span>
        </div>
        <button
          onClick={() => setHintDrawerOpen(false)}
          className="text-amber-400/60 hover:text-amber-300 p-1 rounded-lg hover:bg-amber-500/10 transition-colors"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <p className="mt-2 text-sm text-amber-100/90 leading-relaxed">
        {currentDecision.question.hint}
      </p>

      <div className="mt-3 text-[11px] text-amber-400/70 font-mono">
        Pedagogical guidance: BKT transition boost P(T) + 0.10 applied when utilizing hints.
      </div>
    </div>
  );
};
