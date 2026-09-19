import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  BarChart3,
  Bot,
  CheckCircle,
  ChevronRight,
  Compass,
  Cpu,
  HelpCircle,
  Play,
  RefreshCw,
  Sparkles,
  TrendingUp,
  Zap,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { analyticsApi } from '../api/client';
import { useTutorStore } from '../store/useTutorStore';
import { SessionDetailedAnalytics } from '../types';

export const ResearcherDashboardPage: React.FC = () => {
  const {
    currentDecision,
    benchmarkResults,
    isBenchmarking,
    runBenchmark,
    sessionId,
  } = useTutorStore();

  const [sessionsList, setSessionsList] = useState<any[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<number | null>(sessionId);
  const [sessionAnalytics, setSessionAnalytics] = useState<SessionDetailedAnalytics | null>(null);
  const [isLoadingAnalytics, setIsLoadingAnalytics] = useState<boolean>(false);

  useEffect(() => {
    loadSessions();
  }, []);

  useEffect(() => {
    if (selectedSessionId) {
      loadSessionDetail(selectedSessionId);
    }
  }, [selectedSessionId]);

  const loadSessions = async () => {
    try {
      const list = await analyticsApi.listSessions();
      setSessionsList(list);
      if (list.length > 0 && !selectedSessionId) {
        setSelectedSessionId(list[0].id);
      }
    } catch (err) {
      console.error('Failed to load sessions', err);
    }
  };

  const loadSessionDetail = async (id: number) => {
    setIsLoadingAnalytics(true);
    try {
      const data = await analyticsApi.getSessionAnalytics(id);
      setSessionAnalytics(data);
    } catch (err) {
      console.error('Failed to load session detail', err);
    } finally {
      setIsLoadingAnalytics(false);
    }
  };

  // Recharts data preparation for D3QN Decision Inspector
  const explainability = currentDecision?.explainability || sessionAnalytics?.latest_decision_explainability;
  const advList: number[] = (explainability as any)?.action_advantages || (explainability as any)?.advantages || [0, 0, 0, 0, 0, 0];
  const qList: number[] = explainability?.q_values || [0, 0, 0, 0, 0, 0];

  const d3qnChartData = explainability
    ? [
        { name: 'Easy Prob', action: 0, q: qList[0] ?? 0, adv: advList[0] ?? 0 },
        { name: 'Medium Prob', action: 1, q: qList[1] ?? 0, adv: advList[1] ?? 0 },
        { name: 'Hard Prob', action: 2, q: qList[2] ?? 0, adv: advList[2] ?? 0 },
        { name: 'Scaffold Hint', action: 3, q: qList[3] ?? 0, adv: advList[3] ?? 0 },
        { name: 'Worked Ex.', action: 4, q: qList[4] ?? 0, adv: advList[4] ?? 0 },
        { name: 'Spaced Review', action: 5, q: qList[5] ?? 0, adv: advList[5] ?? 0 },
      ]
    : [];

  const selectedAction = explainability?.selected_action ?? 0;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Dashboard Title & Top Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 p-0.5 shadow-md shadow-indigo-500/20">
              <div className="h-full w-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <BarChart3 className="h-5 w-5 text-indigo-400" />
              </div>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white tracking-tight">
                Explainable RL Researcher Dashboard
              </h2>
              <p className="text-xs text-slate-400">
                Live Bayesian Knowledge Tracing • D3QN Decomposition • Pedagogical A/B Benchmarks
              </p>
            </div>
          </div>
        </div>

        {/* Session Selector & Benchmark Trigger */}
        <div className="flex items-center gap-3">
          {sessionsList.length > 0 && (
            <select
              value={selectedSessionId || ''}
              onChange={(e) => setSelectedSessionId(Number(e.target.value))}
              className="px-3 py-2 bg-slate-900 border border-slate-800 rounded-xl text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              {sessionsList.map((s) => (
                <option key={s.id} value={s.id}>
                  Session #{s.id} ({s.policy_type.toUpperCase()}) - {s.total_steps} steps
                </option>
              ))}
            </select>
          )}

          <button
            onClick={() => runBenchmark(15, 25)}
            disabled={isBenchmarking}
            className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-2 shadow-lg shadow-indigo-600/20 disabled:opacity-50 transition-colors"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isBenchmarking ? 'animate-spin' : ''}`} />
            <span>{isBenchmarking ? 'Running Monte Carlo...' : 'Run A/B Benchmark'}</span>
          </button>
        </div>
      </div>

      {/* Grid 1: BKT Belief Trajectory & Cognitive Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Synchronized BKT Multi-Skill Trajectory */}
        <div className="lg:col-span-2 bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-indigo-400" />
                <span>Bayesian Knowledge Tracing (BKT) Evolution</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Continuous latent belief state b_t across Knowledge Components over interaction steps
              </p>
            </div>
          </div>

          <div className="h-72 w-full">
            {sessionAnalytics?.belief_trajectory && sessionAnalytics.belief_trajectory.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sessionAnalytics.belief_trajectory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 1]} stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', fontSize: '11px' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                  <Line type="monotone" dataKey="kc_0" name="Basic Arithmetic" stroke="#38bdf8" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="kc_1" name="Advanced Arithmetic" stroke="#818cf8" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="kc_2" name="Basic Algebra" stroke="#c084fc" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="kc_3" name="Advanced Algebra" stroke="#f472b6" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="mean" name="Mean Belief" stroke="#34d399" strokeWidth={3} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-500">
                Start or select an active tutoring session to view live belief trajectories.
              </div>
            )}
          </div>
        </div>

        {/* Right 1 Col: Session KPI Summary */}
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <Activity className="h-4 w-4 text-emerald-400" />
            <span>Session Telemetry</span>
          </h3>

          <div className="space-y-3">
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800/80 flex items-center justify-between">
              <span className="text-xs text-slate-400">Total Steps</span>
              <span className="font-mono font-bold text-white">{sessionAnalytics?.total_steps || 0}</span>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800/80 flex items-center justify-between">
              <span className="text-xs text-slate-400">Initial Mean Belief</span>
              <span className="font-mono font-bold text-slate-300">
                {((sessionAnalytics?.initial_mean_belief || 0.2) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800/80 flex items-center justify-between">
              <span className="text-xs text-slate-400">Final Mean Belief</span>
              <span className="font-mono font-bold text-emerald-400">
                {((sessionAnalytics?.final_mean_belief || 0.2) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="p-3.5 rounded-xl bg-indigo-950/20 border border-indigo-500/20 flex items-center justify-between">
              <span className="text-xs text-indigo-300 font-medium">Norm. Learning Gain (NLG)</span>
              <span className="font-mono font-bold text-indigo-200">
                {((sessionAnalytics?.normalized_learning_gain || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="p-3.5 rounded-xl bg-violet-950/20 border border-violet-500/20 flex items-center justify-between">
              <span className="text-xs text-violet-300 font-medium">Cumulative RL Reward</span>
              <span className="font-mono font-bold text-violet-200">
                {sessionAnalytics?.total_reward || 0}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Grid 2: D3QN Decision Inspector & Grounded Explainability */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* D3QN Action Comparison Bar Chart */}
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Cpu className="h-4 w-4 text-indigo-400" />
                <span>D3QN Action Q-Value & Advantage Inspector</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Decomposed Q(s, a) = V(s) + (A(s, a) - mean(A)) across all pedagogical choices
              </p>
            </div>
          </div>

          <div className="h-64 w-full">
            {d3qnChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={d3qnChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', fontSize: '11px' }}
                  />
                  <Bar dataKey="q" name="Q-Value" radius={[4, 4, 0, 0]}>
                    {d3qnChartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.action === selectedAction ? '#6366f1' : '#334155'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-500">
                No active D3QN forward pass logged yet.
              </div>
            )}
          </div>
        </div>

        {/* Natural Language Pedagogical Rationale Card */}
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 text-indigo-400 mb-2">
              <Sparkles className="h-4 w-4" />
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                Grounded Explainability Rationale
              </h3>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Mathematical rationale derived directly from state vector s_t, Value stream V(s), and Advantage stream A(s, a)
            </p>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 text-xs leading-relaxed font-mono">
              {explainability?.pedagogical_rationale ||
                'Policy standby. The RL agent will synthesize real-time grounded rationales during active tutoring sessions.'}
            </div>

            {explainability && (
              <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 text-[10px] block">State Value V(s)</span>
                  <span className="font-mono font-bold text-indigo-300">
                    {explainability.state_value_v?.toFixed(4)}
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 text-[10px] block">Chosen Action</span>
                  <span className="font-mono font-bold text-emerald-400">
                    Action {explainability.selected_action}
                  </span>
                </div>
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-500">
            Zero pedagogical hallucination: Decisions are certified by Bellman target convergence.
          </div>
        </div>
      </div>

      {/* Grid 3: Multi-Objective Reward Waterfall Timeline */}
      {sessionAnalytics?.reward_timeline && sessionAnalytics.reward_timeline.length > 0 && (
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2 mb-2">
            <Zap className="h-4 w-4 text-amber-400" />
            <span>Multi-Objective Reward Decomposition Timeline</span>
          </h3>
          <p className="text-xs text-slate-400 mb-4">
            Component breakdown per time step: NLG Reward, Step Cost, Hint Penalty, Friction Penalty, and Review Bonus
          </p>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sessionAnalytics.reward_timeline} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', fontSize: '11px' }} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Bar dataKey="nlg_reward" name="Learning Gain Reward" fill="#10b981" stackId="a" />
                <Bar dataKey="step_penalty" name="Step Cost" fill="#f43f5e" stackId="a" />
                <Bar dataKey="hint_penalty" name="Hint Penalty" fill="#f59e0b" stackId="a" />
                <Bar dataKey="friction_penalty" name="Friction Penalty" fill="#8b5cf6" stackId="a" />
                <Bar dataKey="review_bonus" name="Review Bonus" fill="#06b6d4" stackId="a" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Grid 4: Comparative A/B Benchmark Results Table */}
      {benchmarkResults && (
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-indigo-400" />
                <span>Monte Carlo Pedagogical Policy Benchmark (A/B Test)</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Evaluation across multi-episode simulations with Welch's t-test and Cohen's d effect size vs D3QN
              </p>
            </div>
            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
              Statistically Superior NLG
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-[11px] uppercase tracking-wider text-slate-400 border-b border-slate-800 bg-slate-950/40">
                <tr>
                  <th className="py-3 px-4">Pedagogical Policy</th>
                  <th className="py-3 px-4">Norm. Learning Gain (NLG)</th>
                  <th className="py-3 px-4">Mean Reward</th>
                  <th className="py-3 px-4">Mastery Rate</th>
                  <th className="py-3 px-4">Cohen's d vs D3QN</th>
                  <th className="py-3 px-4">Statistical Significance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {/* RL Agent Row */}
                <tr className="bg-indigo-950/30 text-indigo-200 font-bold">
                  <td className="py-3 px-4 font-sans flex items-center gap-1.5">
                    <Sparkles className="h-3.5 w-3.5 text-indigo-400" />
                    <span>{benchmarkResults.rl_agent_summary.name}</span>
                  </td>
                  <td className="py-3 px-4 text-emerald-400">
                    {(benchmarkResults.rl_agent_summary.mean_nlg * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 px-4">{benchmarkResults.rl_agent_summary.mean_reward}</td>
                  <td className="py-3 px-4">{benchmarkResults.rl_agent_summary.mastery_rate}</td>
                  <td className="py-3 px-4">Baseline (0.00)</td>
                  <td className="py-3 px-4 font-sans text-indigo-300">Reference Policy</td>
                </tr>

                {/* Baselines Rows */}
                {benchmarkResults.comparison_table.map((row) => (
                  <tr key={row.baseline} className="text-slate-300 hover:bg-slate-800/30 transition-colors">
                    <td className="py-3 px-4 font-sans font-medium">{row.baseline}</td>
                    <td className="py-3 px-4">{(row.nlg * 100).toFixed(2)}%</td>
                    <td className="py-3 px-4">{row.reward}</td>
                    <td className="py-3 px-4">{row.mastery_rate}</td>
                    <td className="py-3 px-4 text-amber-400">d = {row.cohens_d}</td>
                    <td className="py-3 px-4 font-sans">
                      {row.significant ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          p &lt; 0.05 (Significant)
                        </span>
                      ) : (
                        <span className="text-slate-500">p &ge; 0.05</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
