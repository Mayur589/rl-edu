"""Analytics & Telemetry Endpoints for Researchers and Supervisors."""

import json
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.db.models import (
    D3QNDecisionLog,
    InteractionSession,
    KCBeliefSnapshot,
    RewardLog,
    StepLog,
    StudentProfile,
    User,
)
from app.schemas.analytics import (
    BeliefTrajectoryPoint,
    RewardTimelinePoint,
    SessionDetailedAnalytics,
    StepTelemetryPoint,
)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/sessions", response_model=List[Dict[str, Any]])
def list_sessions(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Lists interaction sessions (user's own if student, all if researcher)."""
    query = db.query(InteractionSession)
    if current_user.role != "researcher":
        query = query.filter(InteractionSession.user_id == current_user.id)

    sessions = query.order_by(InteractionSession.start_time.desc()).limit(50).all()
    results = []
    for s in sessions:
        results.append({
            "id": s.id,
            "user_id": s.user_id,
            "username": s.user.username if s.user else "Unknown",
            "policy_type": s.policy_type,
            "status": s.status,
            "total_steps": s.total_steps,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "initial_mean_belief": round(s.initial_mean_belief, 3),
            "final_mean_belief": round(s.final_mean_belief, 3),
            "normalized_learning_gain": round(s.normalized_learning_gain, 3),
            "total_reward": round(s.total_reward, 3),
        })
    return results


@router.get("/sessions/{session_id}", response_model=SessionDetailedAnalytics)
def get_session_analytics(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieves full synchronized telemetry, belief trajectory, and reward breakdown for Recharts."""
    sess = db.query(InteractionSession).filter(InteractionSession.id == session_id).first()
    if not sess:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if current_user.role != "researcher" and sess.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    # Belief snapshots
    snaps = (
        db.query(KCBeliefSnapshot)
        .filter(KCBeliefSnapshot.session_id == session_id)
        .order_by(KCBeliefSnapshot.step_number.asc())
        .all()
    )
    trajectory = [
        BeliefTrajectoryPoint(
            step=s.step_number,
            kc_0=round(s.kc_0_belief, 4),
            kc_1=round(s.kc_1_belief, 4),
            kc_2=round(s.kc_2_belief, 4),
            kc_3=round(s.kc_3_belief, 4),
            mean=round(s.mean_belief, 4),
            urgency=round(s.review_urgency, 4),
            timestamp=s.timestamp,
        )
        for s in snaps
    ]

    # Step telemetry
    steps = (
        db.query(StepLog)
        .filter(StepLog.session_id == session_id)
        .order_by(StepLog.step_number.asc())
        .all()
    )
    step_telemetry = [
        StepTelemetryPoint(
            step=st.step_number,
            question_id=st.question_id,
            action_name=st.action_name,
            is_correct=st.is_correct,
            response_time=round(st.response_time_seconds, 2),
            friction=round(st.friction_index, 3),
            hint_requested=st.hint_requested,
        )
        for st in steps
    ]

    # Reward logs
    rewards = (
        db.query(RewardLog)
        .filter(RewardLog.session_id == session_id)
        .order_by(RewardLog.step_number.asc())
        .all()
    )
    reward_timeline = [
        RewardTimelinePoint(
            step=r.step_number,
            nlg_reward=round(r.nlg_reward, 3),
            step_penalty=round(r.step_penalty, 3),
            hint_penalty=round(r.hint_penalty, 3),
            friction_penalty=round(r.friction_penalty, 3),
            review_bonus=round(r.review_bonus, 3),
            mastery_bonus=round(r.mastery_bonus, 3),
            total_reward=round(r.total_reward, 3),
        )
        for r in rewards
    ]

    # Latest D3QN decision explainability
    latest_decision = (
        db.query(D3QNDecisionLog)
        .filter(D3QNDecisionLog.session_id == session_id)
        .order_by(D3QNDecisionLog.step_number.desc())
        .first()
    )
    decision_dict = None
    if latest_decision:
        decision_dict = {
            "step_number": latest_decision.step_number,
            "state_value_v": round(latest_decision.state_value_v, 4),
            "q_values": json.loads(latest_decision.q_values_json),
            "advantages": json.loads(latest_decision.advantages_json),
            "selected_action": latest_decision.selected_action,
            "pedagogical_rationale": latest_decision.pedagogical_rationale,
        }

    return SessionDetailedAnalytics(
        session_id=sess.id,
        user_id=sess.user_id,
        username=sess.user.username if sess.user else "Unknown",
        policy_type=sess.policy_type,
        status=sess.status,
        total_steps=sess.total_steps,
        start_time=sess.start_time,
        end_time=sess.end_time,
        initial_mean_belief=round(sess.initial_mean_belief, 4),
        final_mean_belief=round(sess.final_mean_belief, 4),
        normalized_learning_gain=round(sess.normalized_learning_gain, 4),
        total_reward=round(sess.total_reward, 4),
        belief_trajectory=trajectory,
        step_telemetry=step_telemetry,
        reward_timeline=reward_timeline,
        latest_decision_explainability=decision_dict,
    )


@router.get("/dashboard/stats")
def get_global_dashboard_stats(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Returns global platform metrics for researcher dashboard overview."""
    total_students = db.query(User).filter(User.role == "student").count()
    total_sessions = db.query(InteractionSession).count()
    total_steps = db.query(StepLog).count()

    profiles = db.query(StudentProfile).all()
    avg_mastery = float(sum(p.mean_mastery for p in profiles) / len(profiles)) if profiles else 0.20

    completed_sessions = (
        db.query(InteractionSession).filter(InteractionSession.status == "completed").all()
    )
    avg_nlg = (
        float(sum(s.normalized_learning_gain for s in completed_sessions) / len(completed_sessions))
        if completed_sessions
        else 0.0
    )

    return {
        "total_students": total_students,
        "total_sessions": total_sessions,
        "total_steps": total_steps,
        "average_student_mastery": round(avg_mastery, 4),
        "average_normalized_learning_gain": round(avg_nlg, 4),
    }
