/**
 * Comprehensive TypeScript Type Definitions for RL Tutor Platform.
 */

export type UserRole = 'student' | 'researcher';

export interface UserProfile {
  current_streak: number;
  total_sessions: number;
  total_questions_solved: number;
  mean_mastery: number;
  kc_0_belief: number;
  kc_1_belief: number;
  kc_2_belief: number;
  kc_3_belief: number;
}

export interface User {
  id: number;
  username: string;
  email: string;
  role: UserRole;
  profile?: UserProfile;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  role: UserRole;
  username: string;
  user_id: number;
}

export interface KnowledgeComponent {
  idx: number;
  name: string;
  description: string;
  prerequisites: number[];
}

export interface QuestionPublic {
  id: string;
  kc_idx: number;
  kc_name: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  difficulty_val: number;
  prompt: string;
  hint: string;
  worked_example_prompt: string;
  worked_example_explanation: string;
}

export interface ActionRankingItem {
  rank: number;
  action: number;
  name: string;
  q_value: number;
  advantage: number;
  is_selected: boolean;
}

export interface ExplainabilityPayload {
  state_vector: number[];
  state_value_v: number;
  action_advantages: number[];
  q_values: number[];
  selected_action: number;
  selected_action_name: string;
  action_ranking: ActionRankingItem[];
  pedagogical_rationale: string;
}

export interface TutorDecision {
  session_id: number;
  step_number: number;
  action_idx: number;
  action_name: string;
  target_kc_idx: number;
  target_kc_name: string;
  is_review_session: boolean;
  question: QuestionPublic;
  beliefs: number[];
  mean_belief: number;
  review_urgency: number;
  explainability: ExplainabilityPayload;
}

export interface SubmitAnswerResult {
  session_id: number;
  step_number: number;
  is_correct: boolean;
  student_answer: number;
  correct_answer: number;
  explanation: string;
  new_beliefs: number[];
  new_mean_belief: number;
  learning_gain: number;
  reward_breakdown: {
    nlg_reward: number;
    step_penalty: number;
    hint_penalty: number;
    friction_penalty: number;
    review_bonus: number;
    mastery_bonus: number;
    total_reward: number;
  };
  is_session_completed: boolean;
}

export interface BeliefTrajectoryPoint {
  step: number;
  kc_0: number;
  kc_1: number;
  kc_2: number;
  kc_3: number;
  mean: number;
  urgency: number;
  timestamp: string;
}

export interface StepTelemetryPoint {
  step: number;
  question_id: string;
  action_name: string;
  is_correct: boolean;
  response_time: number;
  friction: number;
  hint_requested: boolean;
}

export interface RewardTimelinePoint {
  step: number;
  nlg_reward: number;
  step_penalty: number;
  hint_penalty: number;
  friction_penalty: number;
  review_bonus: number;
  mastery_bonus: number;
  total_reward: number;
}

export interface SessionDetailedAnalytics {
  session_id: number;
  user_id: number;
  username: string;
  policy_type: string;
  status: string;
  total_steps: number;
  start_time: string;
  end_time?: string;
  initial_mean_belief: number;
  final_mean_belief: number;
  normalized_learning_gain: number;
  total_reward: number;
  belief_trajectory: BeliefTrajectoryPoint[];
  step_telemetry: StepTelemetryPoint[];
  reward_timeline: RewardTimelinePoint[];
  latest_decision_explainability?: {
    step_number: number;
    state_value_v: number;
    q_values: number[];
    advantages: number[];
    selected_action: number;
    pedagogical_rationale: string;
  };
}

export interface BenchmarkComparison {
  baseline: string;
  nlg: number;
  reward: number;
  mastery_rate: string;
  cohens_d: number;
  significant: boolean;
}

export interface BenchmarkResponse {
  rl_agent_summary: {
    name: string;
    mean_nlg: number;
    mean_reward: number;
    mastery_rate: string;
  };
  all_policies: any[];
  comparison_table: BenchmarkComparison[];
}
