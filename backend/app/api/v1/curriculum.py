"""Curriculum and Question Exploration Endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.domain.curriculum.repository import (
    QUESTION_BANK,
    get_all_kcs,
    get_kc_by_idx,
    get_question_by_id,
    get_questions_by_kc,
)
from app.schemas.curriculum import KnowledgeComponentResponse, QuestionResponse

router = APIRouter(prefix="/curriculum", tags=["Curriculum"])


@router.get("/kcs", response_model=List[KnowledgeComponentResponse])
def list_knowledge_components():
    """Retrieves all 4 Knowledge Components with descriptions and prerequisite DAG."""
    kcs = get_all_kcs()
    return [
        KnowledgeComponentResponse(
            idx=kc.idx,
            name=kc.name,
            description=kc.description,
            prerequisites=kc.prerequisites,
        )
        for kc in kcs
    ]


@router.get("/questions", response_model=List[QuestionResponse])
def list_questions(
    kc_idx: Optional[int] = Query(None, ge=0, le=3, description="Filter by KC index (0-3)"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty ('Easy', 'Medium', 'Hard')"),
):
    """Retrieves questions from the curated 84-question psychometric bank."""
    results: List[QuestionResponse] = []
    if kc_idx is not None:
        items = get_questions_by_kc(kc_idx, difficulty)
    else:
        items = []
        for k in QUESTION_BANK.keys():
            items.extend(get_questions_by_kc(k, difficulty))

    for q in items:
        results.append(
            QuestionResponse(
                id=q.id,
                kc_idx=q.kc_idx,
                kc_name=q.kc_name,
                difficulty=q.difficulty,
                difficulty_val=q.difficulty_val,
                prompt=q.prompt,
                explanation=q.explanation,
                hint=q.hint,
                worked_example_prompt=q.worked_example_prompt,
                worked_example_explanation=q.worked_example_explanation,
                is_review_eligible=q.is_review_eligible,
            )
        )
    return results


@router.get("/questions/{question_id}", response_model=QuestionResponse)
def get_question(question_id: str):
    """Retrieves a specific question item by ID."""
    q = get_question_by_id(question_id)
    if not q:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Question '{question_id}' not found")
    return QuestionResponse(
        id=q.id,
        kc_idx=q.kc_idx,
        kc_name=q.kc_name,
        difficulty=q.difficulty,
        difficulty_val=q.difficulty_val,
        prompt=q.prompt,
        explanation=q.explanation,
        hint=q.hint,
        worked_example_prompt=q.worked_example_prompt,
        worked_example_explanation=q.worked_example_explanation,
        is_review_eligible=q.is_review_eligible,
    )
