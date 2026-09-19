"""Pydantic Schemas for Tutor Decisions, Interactive Steps, and Explainability."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.curriculum import QuestionPublicResponse, QuestionResponse


class StartSessionRequest(BaseModel):
    policy_type: Optional[str] = Field("d3qn", pattern="^(d3qn|heuristic|linear|random)$")


class StartSessionResponse(BaseModel):
    session_id: int
    policy_type: str
    status: str
    initial_beliefs: List[float]
    mean_belief: float
    start_time: datetime


class ExplainabilityPayload(BaseModel):
    state_vector: List[float]
    state_value_v: float
    action_advantages: List[float]
    q_values: List[float]
    selected_action: int
    selected_action_name: str
    action_ranking: List[Dict[str, Any]]
    pedagogical_rationale: str


class TutorDecisionResponse(BaseModel):
    session_id: int
    step_number: int
    action_idx: int
    action_name: str
    target_kc_idx: int
    target_kc_name: str
    is_review_session: bool
    question: Optional[QuestionPublicResponse] = None
    beliefs: List[float]
    mean_belief: float
    review_urgency: float
    explainability: Optional[ExplainabilityPayload] = None


class SubmitAnswerRequest(BaseModel):
    question_id: str
    student_answer: float
    response_time_seconds: float = Field(..., ge=0.0)
    hint_viewed: bool = False
    worked_example_viewed: bool = False


class SubmitAnswerResponse(BaseModel):
    session_id: int
    step_number: int
    is_correct: bool
    student_answer: float
    correct_answer: float
    explanation: str
    new_beliefs: List[float]
    new_mean_belief: float
    learning_gain: float
    reward_breakdown: Dict[str, float]
    is_session_completed: bool
    next_decision: Optional[TutorDecisionResponse] = None
