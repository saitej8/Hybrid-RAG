"""Authentication: register / login backed by SQLite + bcrypt."""
from .auth_manager import (
    AuthError,
    UserExistsError,
    InvalidCredentials,
    WeakPasswordError,
    InvalidUsernameError,
    AuthManager,
)

__all__ = [
    "AuthError",
    "UserExistsError",
    "InvalidCredentials",
    "WeakPasswordError",
    "InvalidUsernameError",
    "AuthManager",
]
