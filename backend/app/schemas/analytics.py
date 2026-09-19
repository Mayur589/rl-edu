"""Pydantic Schemas for Analytics, Telemetry, and Benchmarks."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class BeliefTrajectoryPoint(BaseModel):
    step: int
    kc_0: float
    kc_1: float
    kc_2: float
    kc_3: float
    mean: float
    urgency: float
    timestamp: datetime


class StepTelemetryPoint(BaseModel):
    step: int
    question_id: str
    action_name: str
    is_correct: bool
    response_time: float
    friction: float
    hint_requested: bool


class RewardTimelinePoint(BaseModel):
    step: int
    nlg_reward: float
    step_penalty: float
    hint_penalty: float
    friction_penalty: float
    review_bonus: float
    mastery_bonus: float
    total_reward: float


class SessionDetailedAnalytics(BaseModel):
    session_id: int
    user_id: int
    username: str
    policy_type: str
    status: str
    total_steps: int
    start_time: datetime
    end_time: Optional[datetime]
    initial_mean_belief: float
    final_mean_belief: float
    normalized_learning_gain: float
    total_reward: float
    belief_trajectory: List[BeliefTrajectoryPoint]
    step_telemetry: List[StepTelemetryPoint]
    reward_timeline: List[RewardTimelinePoint]
    latest_decision_explainability: Optional[Dict[str, Any]] = None


class BenchmarkRunRequest(BaseModel):
    episodes: int = 15
    max_steps: int = 25


class BenchmarkPolicyComparison(BaseModel):
    baseline: str
    nlg: float
    reward: float
    mastery_rate: str
    cohens_d: float
    significant: bool


class BenchmarkResultResponse(BaseModel):
    rl_agent_summary: Dict[str, Any]
    all_policies: List[Dict[str, Any]]
    comparison_table: List[BenchmarkPolicyComparison]
