"""Pydantic Schemas for Authentication and User Management."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: Optional[str] = Field("student", pattern="^(student|researcher)$")


class UserLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    user_id: int


class StudentProfileResponse(BaseModel):
    current_streak: int
    total_sessions: int
    total_questions_solved: int
    mean_mastery: float
    kc_0_belief: float
    kc_1_belief: float
    kc_2_belief: float
    kc_3_belief: float


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    profile: Optional[StudentProfileResponse] = None
    created_at: datetime

    class Config:
        from_attributes = True
