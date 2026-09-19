"""SQLAlchemy 2.0 ORM Entity Models for RL Tutor Platform."""

from datetime import datetime, timezone
import json
from typing import Any, List, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    """User account entity for students and researchers."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="student", nullable=False)  # 'student' or 'researcher'
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    # Relationships
    profile = relationship("StudentProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    sessions = relationship("InteractionSession", back_populates="user", cascade="all, delete-orphan")
    review_items = relationship("ReviewQueueItem", back_populates="user", cascade="all, delete-orphan")


class StudentProfile(Base):
    """Student longitudinal metrics and mastery summary."""

    __tablename__ = "student_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    current_streak = Column(Integer, default=0, nullable=False)
    total_sessions = Column(Integer, default=0, nullable=False)
    total_questions_solved = Column(Integer, default=0, nullable=False)
    mean_mastery = Column(Float, default=0.20, nullable=False)
    kc_0_belief = Column(Float, default=0.20, nullable=False)
    kc_1_belief = Column(Float, default=0.20, nullable=False)
    kc_2_belief = Column(Float, default=0.20, nullable=False)
    kc_3_belief = Column(Float, default=0.20, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    user = relationship("User", back_populates="profile")


class InteractionSession(Base):
    """A distinct tutoring session episode between student and RL policy."""

    __tablename__ = "interaction_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    policy_type = Column(String(50), default="d3qn", nullable=False)
    status = Column(String(20), default="active", nullable=False)  # 'active', 'completed', 'abandoned'
    start_time = Column(DateTime, default=utcnow, nullable=False)
    end_time = Column(DateTime, nullable=True)
    total_steps = Column(Integer, default=0, nullable=False)
    initial_mean_belief = Column(Float, default=0.20, nullable=False)
    final_mean_belief = Column(Float, default=0.20, nullable=False)
    total_reward = Column(Float, default=0.0, nullable=False)
    normalized_learning_gain = Column(Float, default=0.0, nullable=False)

    user = relationship("User", back_populates="sessions")
    step_logs = relationship("StepLog", back_populates="session", cascade="all, delete-orphan")
    belief_snapshots = relationship("KCBeliefSnapshot", back_populates="session", cascade="all, delete-orphan")
    decision_logs = relationship("D3QNDecisionLog", back_populates="session", cascade="all, delete-orphan")
    reward_logs = relationship("RewardLog", back_populates="session", cascade="all, delete-orphan")


class StepLog(Base):
    """Fine-grained interaction log for each problem presented and answered."""

    __tablename__ = "step_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interaction_sessions.id", ondelete="CASCADE"), nullable=False)
    step_number = Column(Integer, nullable=False)
    question_id = Column(String(50), nullable=False)
    kc_idx = Column(Integer, nullable=False)
    action_idx = Column(Integer, nullable=False)
    action_name = Column(String(50), nullable=False)
    is_correct = Column(Boolean, nullable=False)
    response_time_seconds = Column(Float, default=0.0, nullable=False)
    friction_index = Column(Float, default=0.0, nullable=False)
    hint_requested = Column(Boolean, default=False, nullable=False)
    worked_example_viewed = Column(Boolean, default=False, nullable=False)
    timestamp = Column(DateTime, default=utcnow, nullable=False)

    session = relationship("InteractionSession", back_populates="step_logs")


class KCBeliefSnapshot(Base):
    """Snapshot of BKT belief state at each step of a session."""

    __tablename__ = "kc_belief_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interaction_sessions.id", ondelete="CASCADE"), nullable=False)
    step_number = Column(Integer, nullable=False)
    kc_0_belief = Column(Float, nullable=False)
    kc_1_belief = Column(Float, nullable=False)
    kc_2_belief = Column(Float, nullable=False)
    kc_3_belief = Column(Float, nullable=False)
    mean_belief = Column(Float, nullable=False)
    review_urgency = Column(Float, default=0.0, nullable=False)
    timestamp = Column(DateTime, default=utcnow, nullable=False)

    session = relationship("InteractionSession", back_populates="belief_snapshots")


class D3QNDecisionLog(Base):
    """Explainability record capturing exact V(s), A(s, a), Q(s, a), and rationale."""

    __tablename__ = "d3qn_decision_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interaction_sessions.id", ondelete="CASCADE"), nullable=False)
    step_number = Column(Integer, nullable=False)
    state_vector_json = Column(Text, nullable=False)
    state_value_v = Column(Float, nullable=False)
    q_values_json = Column(Text, nullable=False)
    advantages_json = Column(Text, nullable=False)
    selected_action = Column(Integer, nullable=False)
    pedagogical_rationale = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=utcnow, nullable=False)

    session = relationship("InteractionSession", back_populates="decision_logs")


class RewardLog(Base):
    """Multi-objective reward decomposition log per interaction step."""

    __tablename__ = "reward_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interaction_sessions.id", ondelete="CASCADE"), nullable=False)
    step_number = Column(Integer, nullable=False)
    nlg_reward = Column(Float, nullable=False)
    step_penalty = Column(Float, nullable=False)
    hint_penalty = Column(Float, nullable=False)
    friction_penalty = Column(Float, nullable=False)
    review_bonus = Column(Float, nullable=False)
    mastery_bonus = Column(Float, nullable=False)
    total_reward = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utcnow, nullable=False)

    session = relationship("InteractionSession", back_populates="reward_logs")


class ReviewQueueItem(Base):
    """Spaced repetition schedule item for student retention maintenance."""

    __tablename__ = "review_queue_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    kc_idx = Column(Integer, nullable=False)
    urgency_score = Column(Float, default=0.0, nullable=False)
    last_practiced_at = Column(DateTime, default=utcnow, nullable=False)
    scheduled_for = Column(DateTime, default=utcnow, nullable=False)
    is_completed = Column(Boolean, default=False, nullable=False)

    user = relationship("User", back_populates="review_items")
