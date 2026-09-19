"""API Version 1 Router Aggregation."""

from fastapi import APIRouter

from app.api.v1.analytics import router as analytics_router
from app.api.v1.auth import router as auth_router
from app.api.v1.benchmarks import router as benchmarks_router
from app.api.v1.curriculum import router as curriculum_router
from app.api.v1.tutor import router as tutor_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(curriculum_router)
api_router.include_router(tutor_router)
api_router.include_router(analytics_router)
api_router.include_router(benchmarks_router)
