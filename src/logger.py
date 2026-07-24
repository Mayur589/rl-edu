"""Session Data Logging for A/B Testing and Student Interactions.

This module logs student interactions, response times, tutor actions, and BKT belief trajectories
to `data/session_logs.csv` for offline experimental analysis.
"""

import csv
from datetime import datetime
import os
from typing import Any, Dict, List, Optional


LOG_FILE_PATH = os.path.join("data", "session_logs.csv")
SURVEY_LOG_PATH = os.path.join("data", "affective_surveys.csv")

CSV_HEADERS: List[str] = [
    "timestamp",
    "student_id",
    "tutor_mode",
    "step",
    "kc_idx",
    "kc_name",
    "action_idx",
    "action_name",
    "is_correct",
    "response_time_sec",
    "mean_belief",
    "b_basic_arithmetic",
    "b_adv_arithmetic",
    "b_basic_algebra",
    "b_adv_algebra",
]

SURVEY_HEADERS: List[str] = [
    "timestamp",
    "student_id",
    "tutor_mode",
    "engagement_score",
    "frustration_score",
    "pacing_feedback",
    "comments",
]


def ensure_log_files_exist() -> None:
    """Ensures `data/` directory and CSV log files exist with proper headers."""
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)

    if not os.path.exists(SURVEY_LOG_PATH):
        with open(SURVEY_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(SURVEY_HEADERS)


def log_interaction(
    student_id: str,
    tutor_mode: str,
    step: int,
    kc_idx: int,
    kc_name: str,
    action_idx: int,
    action_name: str,
    is_correct: bool,
    response_time_sec: float,
    mean_belief: float,
    beliefs: List[float],
) -> None:
    """Appends an interaction record to `data/session_logs.csv`.

    Args:
        student_id: Unique identifier for student.
        tutor_mode: Tutor mode ('Control Mode' or 'Experimental Mode').
        step: Current step number in episode.
        kc_idx: Targeted Knowledge Component index.
        kc_name: Targeted Knowledge Component name.
        action_idx: Selected action index.
        action_name: Selected action name.
        is_correct: Whether student answered correctly.
        response_time_sec: Time taken to answer in seconds.
        mean_belief: Overall mean skill belief.
        beliefs: List of 4 skill belief values.
    """
    ensure_log_files_exist()
    timestamp_str = datetime.now().isoformat()

    row = [
        timestamp_str,
        student_id,
        tutor_mode,
        step,
        kc_idx,
        kc_name,
        action_idx,
        action_name,
        1 if is_correct else 0,
        round(response_time_sec, 2),
        round(mean_belief, 4),
        round(beliefs[0], 4) if len(beliefs) > 0 else 0.0,
        round(beliefs[1], 4) if len(beliefs) > 1 else 0.0,
        round(beliefs[2], 4) if len(beliefs) > 2 else 0.0,
        round(beliefs[3], 4) if len(beliefs) > 3 else 0.0,
    ]

    with open(LOG_FILE_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def log_affective_survey(
    student_id: str,
    tutor_mode: str,
    engagement_score: int,
    frustration_score: int,
    pacing_feedback: str,
    comments: str = "",
) -> None:
    """Appends a post-session affective survey record to `data/affective_surveys.csv`.

    Args:
        student_id: Unique student ID.
        tutor_mode: Tutor mode ('Control Mode' or 'Experimental Mode').
        engagement_score: Self-reported engagement (1 to 5).
        frustration_score: Self-reported frustration (1 to 5).
        pacing_feedback: Feedback on pacing ('Too Fast', 'Just Right', 'Too Slow').
        comments: Optional freeform text comments.
    """
    ensure_log_files_exist()
    timestamp_str = datetime.now().isoformat()

    row = [
        timestamp_str,
        student_id,
        tutor_mode,
        engagement_score,
        frustration_score,
        pacing_feedback,
        comments.replace("\n", " "),
    ]

    with open(SURVEY_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
