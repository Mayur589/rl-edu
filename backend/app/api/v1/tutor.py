"""Core Intelligent Tutoring REST Endpoints.

Orchestrates pedagogical decision-making, Bayesian belief updating, question delivery,
real-time explainability, and multi-objective reinforcement learning updates.
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException, status
import numpy as np
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.db.models import (
    D3QNDecisionLog,
    InteractionSession,
    KCBeliefSnapshot,
    RewardLog,
    StepLog,
    StudentProfile,
    User,
)
from app.domain.cognitive.bkt import BKTEngine
from app.domain.curriculum.models import QuestionItem
from app.domain.curriculum.repository import (
    KC_NAMES,
    QUESTION_BANK,
    check_prerequisites_met,
    get_question_by_id,
    get_questions_by_kc,
)
from app.domain.rl.agent import D3QNAgent
from app.domain.rl.baselines import (
    LeitnerSpacedRepetitionPolicy,
    LinearCurriculumPolicy,
    RandomPolicy,
    RuleBasedHeuristicPolicy,
)
from app.domain.rl.env import POMDPTutorEnv
from app.domain.rl.reward import RewardCalculator
from app.schemas.curriculum import QuestionPublicResponse
from app.schemas.tutor import (
    ExplainabilityPayload,
    StartSessionRequest,
    StartSessionResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    TutorDecisionResponse,
)

router = APIRouter(prefix="/tutor", tags=["Intelligent Tutoring System"])

# Singleton RL agent instance serving live session decisions and online learning
GLOBAL_D3QN_AGENT = D3QNAgent(seed=42)
REWARD_CALC = RewardCalculator()


class SessionState:
    """In-memory active state for a live tutoring session."""

    def __init__(
        self,
        session_id: int,
        user_id: int,
        policy_type: str = "d3qn",
        max_steps: int = 25,
        initial_beliefs: Optional[List[float]] = None,
    ) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.policy_type = policy_type
        self.max_steps = max_steps
        self.bkt_engine = BKTEngine()
        if initial_beliefs:
            self.bkt_engine.reset(initial_beliefs)

        self.current_step = 0
        self.recent_history: List[float] = []
        self.last_action: int = 0
        self.last_friction: float = 0.0
        self.consecutive_hints: int = 0
        self.current_question: Optional[QuestionItem] = None
        self.last_obs: Optional[np.ndarray] = None
        self.asked_question_ids: set = set()

    def get_obs(self) -> np.ndarray:
        beliefs = self.bkt_engine.get_beliefs()
        mean_belief = self.bkt_engine.get_mean_belief()
        rolling_acc = float(np.mean(self.recent_history)) if self.recent_history else 0.5
        step_ratio = float(self.current_step / max(1, self.max_steps))
        last_action_norm = float(self.last_action / 5.0)
        urgency = self.bkt_engine.get_review_urgency()

        obs = np.array(
            [
                beliefs[0], beliefs[1], beliefs[2], beliefs[3],
                mean_belief,
                rolling_acc,
                self.last_friction,
                step_ratio,
                last_action_norm,
                urgency,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)


# In-memory tracking for active sessions
ACTIVE_SESSIONS: Dict[int, SessionState] = {}


def select_policy_action(policy_type: str, state: np.ndarray) -> int:
    """Selects pedagogical action given policy type."""
    if policy_type == "heuristic":
        return RuleBasedHeuristicPolicy().select_action(state)
    elif policy_type == "linear":
        return LinearCurriculumPolicy().select_action(state)
    elif policy_type == "random":
        return RandomPolicy().select_action(state)
    else:
        # Default to explainable D3QN agent
        return GLOBAL_D3QN_AGENT.select_action(state, evaluate=False)


def pick_question_for_action(
    session_state: SessionState, action_idx: int
) -> Tuple[QuestionItem, int]:
    """Selects an unasked question matching the pedagogical action and KC priority."""
    beliefs = session_state.bkt_engine.get_beliefs()

    # Determine target KC
    if action_idx == 5:  # Spaced review
        steps = session_state.bkt_engine.steps_since_touch
        target_kc = int(np.argmax([s * (b > 0.3) for s, b in zip(steps, beliefs)]))
    else:
        target_kc = 0
        for k in range(4):
            if beliefs[k] < 0.88 and check_prerequisites_met(k, beliefs.tolist(), threshold=0.65):
                target_kc = k
                break

    # Determine difficulty level from action
    if action_idx == 0:
        diff_label = "Easy"
    elif action_idx == 1:
        diff_label = "Medium"
    elif action_idx == 2:
        diff_label = "Hard"
    elif action_idx == 5:
        diff_label = "Medium"  # Reviews are balanced at Medium
    else:
        # Hint / Worked example scaffolds current KC at Easy or Medium
        diff_label = "Medium" if beliefs[target_kc] > 0.4 else "Easy"

    candidates = get_questions_by_kc(target_kc, diff_label)
    unasked = [q for q in candidates if q.id not in session_state.asked_question_ids]
    if not unasked:
        # If all asked in this difficulty, fall back to any unasked in KC
        all_kc = get_questions_by_kc(target_kc)
        unasked = [q for q in all_kc if q.id not in session_state.asked_question_ids]
        if not unasked:
            unasked = candidates

    selected_q = unasked[0]
    session_state.asked_question_ids.add(selected_q.id)
    return selected_q, target_kc


@router.post("/sessions/start", response_model=StartSessionResponse)
def start_session(
    request: StartSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Initializes a new adaptive tutoring session."""
    profile = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
    initial_beliefs = [0.20, 0.20, 0.20, 0.20]
    if profile:
        initial_beliefs = [
            profile.kc_0_belief,
            profile.kc_1_belief,
            profile.kc_2_belief,
            profile.kc_3_belief,
        ]

    db_session = InteractionSession(
        user_id=current_user.id,
        policy_type=request.policy_type or "d3qn",
        status="active",
        total_steps=0,
        initial_mean_belief=float(np.mean(initial_beliefs)),
        final_mean_belief=float(np.mean(initial_beliefs)),
        total_reward=0.0,
        normalized_learning_gain=0.0,
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    # Instantiate active session state
    session_state = SessionState(
        session_id=db_session.id,
        user_id=current_user.id,
        policy_type=db_session.policy_type,
        initial_beliefs=initial_beliefs,
    )
    ACTIVE_SESSIONS[db_session.id] = session_state

    # Record initial belief snapshot
    snapshot = KCBeliefSnapshot(
        session_id=db_session.id,
        step_number=0,
        kc_0_belief=initial_beliefs[0],
        kc_1_belief=initial_beliefs[1],
        kc_2_belief=initial_beliefs[2],
        kc_3_belief=initial_beliefs[3],
        mean_belief=float(np.mean(initial_beliefs)),
        review_urgency=0.0,
    )
    db.add(snapshot)
    db.commit()

    return StartSessionResponse(
        session_id=db_session.id,
        policy_type=db_session.policy_type,
        status="active",
        initial_beliefs=initial_beliefs,
        mean_belief=float(np.mean(initial_beliefs)),
        start_time=db_session.start_time,
    )


@router.get("/sessions/{session_id}/next-action", response_model=TutorDecisionResponse)
def get_next_action(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Evaluates student cognitive state and returns the next pedagogical recommendation."""
    session_state = ACTIVE_SESSIONS.get(session_id)
    if not session_state:
        # Reconstruct session state from DB if server rebooted
        db_sess = db.query(InteractionSession).filter(InteractionSession.id == session_id).first()
        if not db_sess or db_sess.user_id != current_user.id or db_sess.status != "active":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active session not found")
        last_snap = (
            db.query(KCBeliefSnapshot)
            .filter(KCBeliefSnapshot.session_id == session_id)
            .order_by(KCBeliefSnapshot.step_number.desc())
            .first()
        )
        beliefs = [last_snap.kc_0_belief, last_snap.kc_1_belief, last_snap.kc_2_belief, last_snap.kc_3_belief] if last_snap else [0.2, 0.2, 0.2, 0.2]
        session_state = SessionState(session_id=session_id, user_id=current_user.id, policy_type=db_sess.policy_type, initial_beliefs=beliefs)
        ACTIVE_SESSIONS[session_id] = session_state

    obs = session_state.get_obs()
    session_state.last_obs = obs.copy()

    # 1. RL Policy Action Selection
    action_idx = select_policy_action(session_state.policy_type, obs)
    action_name = POMDPTutorEnv.ACTION_MAP[action_idx]
    session_state.last_action = action_idx

    # 2. Select Question
    question, target_kc = pick_question_for_action(session_state, action_idx)
    session_state.current_question = question

    # 3. Compute Grounded Explainability Decomposition
    explain_record = GLOBAL_D3QN_AGENT.explain_decision(obs)
    explain_dto = ExplainabilityPayload(
        state_vector=explain_record.state_vector,
        state_value_v=explain_record.state_value_v,
        action_advantages=explain_record.action_advantages,
        q_values=explain_record.q_values,
        selected_action=explain_record.selected_action,
        selected_action_name=explain_record.selected_action_name,
        action_ranking=explain_record.action_ranking,
        pedagogical_rationale=explain_record.pedagogical_rationale,
    )

    # 4. Log Decision to DB
    dlog = D3QNDecisionLog(
        session_id=session_id,
        step_number=session_state.current_step + 1,
        state_vector_json=json.dumps(obs.tolist()),
        state_value_v=explain_record.state_value_v,
        q_values_json=json.dumps(explain_record.q_values),
        advantages_json=json.dumps(explain_record.action_advantages),
        selected_action=action_idx,
        pedagogical_rationale=explain_record.pedagogical_rationale,
    )
    db.add(dlog)
    db.commit()

    return TutorDecisionResponse(
        session_id=session_id,
        step_number=session_state.current_step + 1,
        action_idx=action_idx,
        action_name=action_name,
        target_kc_idx=target_kc,
        target_kc_name=KC_NAMES[target_kc],
        is_review_session=bool(action_idx == 5),
        question=QuestionPublicResponse(
            id=question.id,
            kc_idx=question.kc_idx,
            kc_name=question.kc_name,
            difficulty=question.difficulty,
            difficulty_val=question.difficulty_val,
            prompt=question.prompt,
            hint=question.hint,
            worked_example_prompt=question.worked_example_prompt,
            worked_example_explanation=question.worked_example_explanation,
        ),
        beliefs=[round(b, 4) for b in session_state.bkt_engine.get_beliefs().tolist()],
        mean_belief=round(session_state.bkt_engine.get_mean_belief(), 4),
        review_urgency=round(session_state.bkt_engine.get_review_urgency(), 4),
        explainability=explain_dto,
    )


@router.post("/sessions/{session_id}/submit-answer", response_model=SubmitAnswerResponse)
def submit_answer(
    session_id: int,
    request: SubmitAnswerRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Grades student answer, performs BKT belief update, logs reward, and triggers RL learning."""
    session_state = ACTIVE_SESSIONS.get(session_id)
    if not session_state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active session not found")

    question = session_state.current_question or get_question_by_id(request.question_id)
    if not question:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")

    session_state.current_step += 1
    step_num = session_state.current_step

    # 1. Evaluate Correctness
    is_correct = bool(abs(request.student_answer - question.answer) < 0.05)
    correctness_val = 1.0 if is_correct else 0.0
    session_state.recent_history.append(correctness_val)
    if len(session_state.recent_history) > 5:
        session_state.recent_history.pop(0)

    # 2. Cognitive Friction Calculation
    # Normalized response time (capped at 40s) + penalty for mistakes
    norm_time = min(1.0, request.response_time_seconds / 30.0)
    friction = float(np.clip(0.4 * norm_time + (0.5 if not is_correct else 0.0) + (0.1 if request.hint_viewed else 0.0), 0.0, 1.0))
    session_state.last_friction = friction

    # Track consecutive hints
    if request.hint_viewed or request.worked_example_viewed or session_state.last_action in (3, 4):
        session_state.consecutive_hints += 1
    else:
        session_state.consecutive_hints = 0

    prev_mean_belief = session_state.bkt_engine.get_mean_belief()
    is_scaffolding = bool(request.hint_viewed or request.worked_example_viewed or session_state.last_action in (3, 4))

    # 3. BKT Posterior Update
    new_kc_belief = session_state.bkt_engine.update_belief(
        skill_idx=question.kc_idx,
        is_correct=is_correct,
        receives_scaffolding=is_scaffolding,
    )
    new_mean_belief = session_state.bkt_engine.get_mean_belief()
    new_beliefs = session_state.bkt_engine.get_beliefs().tolist()

    # 4. Multi-Objective Reward Calculation
    all_mastered = bool(np.all(np.array(new_beliefs) >= 0.88))
    reward_comp = REWARD_CALC.calculate(
        prev_mean_belief=prev_mean_belief,
        new_mean_belief=new_mean_belief,
        consecutive_hints=session_state.consecutive_hints,
        friction=friction,
        action=session_state.last_action,
        review_urgency=session_state.bkt_engine.get_review_urgency(),
        is_all_mastered=all_mastered,
    )

    # 5. Online RL Agent Update
    next_obs = session_state.get_obs()
    is_completed = all_mastered or (session_state.current_step >= session_state.max_steps)
    if session_state.last_obs is not None:
        GLOBAL_D3QN_AGENT.buffer.push(
            session_state.last_obs,
            session_state.last_action,
            reward_comp.total_reward,
            next_obs,
            is_completed,
        )
        GLOBAL_D3QN_AGENT.train_step()

    # 6. Database Logging
    step_log = StepLog(
        session_id=session_id,
        step_number=step_num,
        question_id=question.id,
        kc_idx=question.kc_idx,
        action_idx=session_state.last_action,
        action_name=POMDPTutorEnv.ACTION_MAP[session_state.last_action],
        is_correct=is_correct,
        response_time_seconds=request.response_time_seconds,
        friction_index=friction,
        hint_requested=request.hint_viewed,
        worked_example_viewed=request.worked_example_viewed,
    )
    db.add(step_log)

    snap = KCBeliefSnapshot(
        session_id=session_id,
        step_number=step_num,
        kc_0_belief=new_beliefs[0],
        kc_1_belief=new_beliefs[1],
        kc_2_belief=new_beliefs[2],
        kc_3_belief=new_beliefs[3],
        mean_belief=new_mean_belief,
        review_urgency=session_state.bkt_engine.get_review_urgency(),
    )
    db.add(snap)

    r_log = RewardLog(
        session_id=session_id,
        step_number=step_num,
        nlg_reward=reward_comp.nlg_reward,
        step_penalty=reward_comp.step_penalty,
        hint_penalty=reward_comp.hint_penalty,
        friction_penalty=reward_comp.friction_penalty,
        review_bonus=reward_comp.review_bonus,
        mastery_bonus=reward_comp.mastery_bonus,
        total_reward=reward_comp.total_reward,
    )
    db.add(r_log)

    # Update session aggregates
    db_session = db.query(InteractionSession).filter(InteractionSession.id == session_id).first()
    if db_session:
        db_session.total_steps = step_num
        db_session.final_mean_belief = new_mean_belief
        db_session.total_reward += reward_comp.total_reward
        denom = max(1e-4, 1.0 - db_session.initial_mean_belief)
        db_session.normalized_learning_gain = (new_mean_belief - db_session.initial_mean_belief) / denom

        if is_completed:
            db_session.status = "completed"
            db_session.end_time = datetime.now(timezone.utc)
            # Remove from active memory cache
            ACTIVE_SESSIONS.pop(session_id, None)

    # Update longitudinal profile
    prof = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
    if prof:
        prof.total_questions_solved += 1
        prof.current_streak = (prof.current_streak + 1) if is_correct else 0
        prof.mean_mastery = new_mean_belief
        prof.kc_0_belief = new_beliefs[0]
        prof.kc_1_belief = new_beliefs[1]
        prof.kc_2_belief = new_beliefs[2]
        prof.kc_3_belief = new_beliefs[3]

    db.commit()

    return SubmitAnswerResponse(
        session_id=session_id,
        step_number=step_num,
        is_correct=is_correct,
        student_answer=request.student_answer,
        correct_answer=question.answer,
        explanation=question.explanation,
        new_beliefs=[round(b, 4) for b in new_beliefs],
        new_mean_belief=round(new_mean_belief, 4),
        learning_gain=round(new_mean_belief - prev_mean_belief, 4),
        reward_breakdown=reward_comp.to_dict(),
        is_session_completed=is_completed,
    )
