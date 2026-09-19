"""Application Configuration Settings.

Uses pydantic-settings to manage environment variables and operational defaults
for database, JWT security, CORS, Bayesian Knowledge Tracing, and RL hyperparameters.
"""

from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings and configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Metadata
    PROJECT_NAME: str = "RL Tutor"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Security & JWT
    SECRET_KEY: str = "super-secret-rl-tutor-key-change-in-production-min-32-chars-long"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    # Defaults to SQLite locally for zero-friction setup, switches to PostgreSQL via env var
    DATABASE_URL: str = "sqlite:///./rl_edu.db"

    # CORS Configuration
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Bayesian Knowledge Tracing (BKT) Defaults
    BKT_P_INIT: float = 0.20
    BKT_P_TRANSIT: float = 0.15
    BKT_P_SLIP: float = 0.10
    BKT_P_GUESS: float = 0.25
    BKT_HINT_BOOST: float = 0.10
    BKT_FORGETTING_RATE: float = 0.02

    # Reinforcement Learning Engine (D3QN) Defaults
    RL_STATE_DIM: int = 10
    RL_ACTION_DIM: int = 6
    RL_HIDDEN_DIM: int = 64
    RL_LEARNING_RATE: float = 0.005
    RL_GAMMA: float = 0.95
    RL_EPSILON_START: float = 1.0
    RL_EPSILON_MIN: float = 0.05
    RL_EPSILON_DECAY: float = 0.995
    RL_BUFFER_CAPACITY: int = 10000
    RL_BATCH_SIZE: int = 32
    RL_TARGET_UPDATE_FREQ: int = 10


settings = Settings()
