"""Security module for password hashing and JWT issuance/validation."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union
import bcrypt
import jwt

from app.core.config import settings


def hash_password(password: str) -> str:
    """Hashes a plaintext password using bcrypt.

    Args:
        password: Raw plaintext password string.

    Returns:
        str: Bcrypt salted password hash.
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a stored bcrypt hash.

    Args:
        plain_password: Raw password to verify.
        hashed_password: Stored bcrypt hash string.

    Returns:
        bool: True if password matches, False otherwise.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def create_access_token(
    subject: Union[str, int],
    role: str = "student",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Creates a signed JWT access token.

    Args:
        subject: User ID or identifier.
        role: User authorization role ('student' or 'researcher').
        expires_delta: Optional custom duration.

    Returns:
        str: Encoded JWT string.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload: Dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    encoded_jwt = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """Decodes and validates a JWT token.

    Args:
        token: JWT string.

    Returns:
        Dict[str, Any]: Decoded payload.

    Raises:
        jwt.PyJWTError: If token is invalid or expired.
    """
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
