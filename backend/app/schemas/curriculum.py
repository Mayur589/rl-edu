"""Pydantic Schemas for Curriculum, Knowledge Components, and Questions."""

from typing import List, Optional
from pydantic import BaseModel


class KnowledgeComponentResponse(BaseModel):
    idx: int
    name: str
    description: str
    prerequisites: List[int]


class QuestionResponse(BaseModel):
    id: str
    kc_idx: int
    kc_name: str
    difficulty: str
    difficulty_val: float
    prompt: str
    explanation: Optional[str] = None
    hint: str
    worked_example_prompt: str
    worked_example_explanation: str
    is_review_eligible: bool


class QuestionPublicResponse(BaseModel):
    """Question schema returned during live tutoring (hides answer and full solution until answered)."""
    id: str
    kc_idx: int
    kc_name: str
    difficulty: str
    difficulty_val: float
    prompt: str
    hint: str
    worked_example_prompt: str
    worked_example_explanation: str
