"""Google OAuth + JWT authentication for the Racing Sheets API.

When GOOGLE_CLIENT_ID is not set, all auth is disabled (local dev mode).
"""
from __future__ import annotations

import os
import time
from typing import Optional

from jose import jwt, JWTError

# ---------------------------------------------------------------------------
# Configuration (all from env vars)
# ---------------------------------------------------------------------------

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-insecure-secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.environ.get("JWT_EXPIRY_HOURS", "24"))
ALLOWED_EMAILS: set[str] = set(
    e.strip().lower()
    for e in os.environ.get("ALLOWED_EMAILS", "").split(",")
    if e.strip()
)
APP_URL = os.environ.get("APP_URL", "http://localhost:8502")

AUTH_ENABLED = bool(GOOGLE_CLIENT_ID)


# ---------------------------------------------------------------------------
# OAuth setup (lazy — only when auth is enabled)
# ---------------------------------------------------------------------------

_oauth = None


def get_oauth():
    """Lazy-init the authlib OAuth registry."""
    global _oauth
    if _oauth is None:
        from authlib.integrations.starlette_client import OAuth

        _oauth = OAuth()
        _oauth.register(
            name="google",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            server_metadata_url=(
                "https://accounts.google.com/.well-known/openid-configuration"
            ),
            client_kwargs={"scope": "openid email profile"},
        )
    return _oauth


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def create_token(email: str, name: str = "") -> str:
    """Mint a JWT for an authenticated user."""
    payload = {
        "sub": email,
        "name": name,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY_HOURS * 3600,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT. Raises JWTError on failure."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def is_email_allowed(email: str) -> bool:
    """Check if an email is on the allowlist (or if no allowlist is set)."""
    if not ALLOWED_EMAILS:
        return True  # no allowlist = allow all authenticated users
    return email.strip().lower() in ALLOWED_EMAILS
