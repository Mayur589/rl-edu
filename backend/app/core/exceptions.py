"""Core domain and application exceptions for RL Tutor."""

from typing import Any, Optional


class RLTutorException(Exception):
    """Base exception class for all RL Tutor errors."""

    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class EntityNotFoundException(RLTutorException):
    """Raised when an expected domain entity or record cannot be found."""
    pass


class InvalidPedagogicalActionException(RLTutorException):
    """Raised when an invalid action index or transition is attempted."""
    pass


class InvalidCognitiveStateException(RLTutorException):
    """Raised when probability bounds or belief calculations violate domain axioms."""
    pass


class AuthenticationFailedException(RLTutorException):
    """Raised when user credentials or JWT verification fails."""
    pass


class SessionClosedException(RLTutorException):
    """Raised when an interaction step is requested on an already terminated session."""
    pass
