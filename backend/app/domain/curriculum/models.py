"""Curriculum domain models and Knowledge Component definitions."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QuestionItem:
    """Represents a psychometrically calibrated curriculum question item.

    Attributes:
        id: Unique identifier string (e.g. 'BA_E01').
        kc_idx: Knowledge Component index in {0, 1, 2, 3}.
        kc_name: Human-readable Knowledge Component name.
        difficulty: Categorical difficulty label ('Easy', 'Medium', 'Hard').
        difficulty_val: Continuous numerical item difficulty beta in [0.0, 1.0] (0.2, 0.5, 0.8).
        prompt: Question text or mathematical expression.
        answer: Target numeric scalar answer.
        explanation: Step-by-step pedagogical solution explanation.
        hint: Scaffolding hint providing guidance without giving away the answer.
        worked_example_prompt: Demonstration prompt for worked example mode.
        worked_example_explanation: Full demonstration solution walkthrough.
        is_review_eligible: Whether this question is suitable for spaced retrieval reviews.
    """

    id: str
    kc_idx: int
    kc_name: str
    difficulty: str
    difficulty_val: float
    prompt: str
    answer: float
    explanation: str
    hint: str
    worked_example_prompt: str
    worked_example_explanation: str
    is_review_eligible: bool = True


@dataclass
class KnowledgeComponent:
    """Represents a Knowledge Component (KC) within the pedagogical taxonomy.

    Attributes:
        idx: Index in {0, 1, 2, 3}.
        name: Name of the KC.
        description: Pedagogical description.
        prerequisites: List of prerequisite KC indices in the learning DAG.
    """

    idx: int
    name: str
    description: str
    prerequisites: List[int]
