"""
auth/db.py — SQLite database layer for RetentionAI authentication.
Stores user accounts with bcrypt-hashed passwords.
"""

import sqlite3
import os
import hashlib
import hmac
import secrets
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "retention_ai.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            email         TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            salt          TEXT    NOT NULL,
            created_at    TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _hash_password(password: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 password hash (no external deps needed)."""
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=260_000,
    )
    return dk.hex()


def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """
    Register a new user.
    Returns (True, "ok") on success, (False, reason) on failure.
    """
    username = username.strip()
    email = email.strip().lower()

    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if "@" not in email:
        return False, "Invalid email address."

    salt = secrets.token_hex(32)
    pw_hash = _hash_password(password, salt)

    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO users (username, email, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, email, pw_hash, salt, datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
        return True, "ok"
    except sqlite3.IntegrityError as e:
        err = str(e)
        if "username" in err:
            return False, "Username already taken."
        if "email" in err:
            return False, "An account with this email already exists."
        return False, "Registration failed."


def authenticate_user(email: str, password: str) -> tuple[bool, dict | str]:
    """
    Authenticate by email + password.
    Returns (True, user_dict) on success, (False, reason) on failure.
    """
    email = email.strip().lower()
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    ).fetchone()
    conn.close()

    if row is None:
        return False, "No account found with this email."

    expected = _hash_password(password, row["salt"])
    if not hmac.compare_digest(expected, row["password_hash"]):
        return False, "Incorrect password."

    return True, {
        "id": row["id"],
        "username": row["username"],
        "email": row["email"],
        "created_at": row["created_at"],
    }


def get_user_by_email(email: str) -> dict | None:
    email = email.strip().lower()
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None
