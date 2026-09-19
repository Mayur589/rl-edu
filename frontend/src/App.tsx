import React, { useEffect, useState } from 'react';
import { AuthModal } from './components/common/AuthModal';
import { Header } from './components/common/Header';
import { ResearcherDashboardPage } from './pages/ResearcherDashboardPage';
import { StudentTutorPage } from './pages/StudentTutorPage';
import { useTutorStore } from './store/useTutorStore';

export const App: React.FC = () => {
  const { isAuthenticated, initAuth } = useTutorStore();
  const [activeTab, setActiveTab] = useState<'student' | 'researcher'>('student');

  useEffect(() => {
    initAuth();
  }, [initAuth]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {!isAuthenticated && <AuthModal />}

      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1">
        {activeTab === 'student' ? <StudentTutorPage /> : <ResearcherDashboardPage />}
      </main>

      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-500">
        RL Tutor Platform • Formulated on Content & Guidance Adaptation (Riedmann et al., 2025)
      </footer>
    </div>
  );
};

export default App;
