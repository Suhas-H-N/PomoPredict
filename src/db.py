"""
Tiny SQLite layer — no ORM needed for an app this size.

Tables:
    users        (id, username, password_hash, created_at)
    predictions  (id, user_id, image_path, top_label, top_confidence,
                  severity, all_predictions_json, lat, lon, created_at,
                  corrected_label)
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "app.db")


@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                image_path TEXT NOT NULL,
                top_label TEXT NOT NULL,
                top_confidence REAL NOT NULL,
                severity TEXT,
                model_version TEXT,
                all_predictions_json TEXT NOT NULL,
                lat REAL,
                lon REAL,
                created_at TEXT NOT NULL,
                corrected_label TEXT
            )
        """)


# ---------------------------------------------------------------- users ----
def create_user(username, password_hash):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def get_user_by_username(username):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


# --------------------------------------------------------- predictions ----
def add_prediction(user_id, image_path, top_label, top_confidence, severity,
                    model_version, all_predictions, lat=None, lon=None):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (user_id, image_path, top_label, top_confidence, severity,
                model_version, all_predictions_json, lat, lon, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, image_path, top_label, top_confidence, severity, model_version,
             json.dumps(all_predictions), lat, lon,
             datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def get_prediction(prediction_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
        return dict(row) if row else None


def list_predictions(user_id=None, limit=200):
    with get_conn() as conn:
        if user_id is None:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]


def set_correction(prediction_id, corrected_label):
    with get_conn() as conn:
        conn.execute(
            "UPDATE predictions SET corrected_label = ? WHERE id = ?",
            (corrected_label, prediction_id),
        )


def list_corrections():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE corrected_label IS NOT NULL"
        ).fetchall()
        return [dict(r) for r in rows]
