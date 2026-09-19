"""FastAPI Main Application Entrypoint for RL Tutor Platform."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import api_router
from app.core.config import settings
from app.core.exceptions import RLTutorException
from app.core.logging import logger
from app.db.session import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown event handler."""
    logger.info("Initializing RL Tutor database schema...")
    Base.metadata.create_all(bind=engine)
    logger.info(f"RL Tutor API started in {settings.ENVIRONMENT} mode.")
    yield
    logger.info("RL Tutor API shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Explainable Reinforcement Learning Adaptive Learning Platform based on Riedmann et al. (2025)",
    lifespan=lifespan,
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RLTutorException)
async def rl_tutor_exception_handler(request: Request, exc: RLTutorException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": exc.__class__.__name__, "message": exc.message, "details": exc.details},
    )


@app.get("/health", tags=["Health"])
def health_check():
    """Service liveness and readiness probe."""
    return {"status": "healthy", "service": settings.PROJECT_NAME, "version": settings.VERSION}


@app.get("/", tags=["Root"])
def root():
    """Root metadata endpoint."""
    return {
        "message": "Welcome to the RL Tutor API",
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR,
    }


# Mount API V1 endpoints
app.include_router(api_router, prefix=settings.API_V1_STR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
