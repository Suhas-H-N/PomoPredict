"""
Minimal username/password auth using Flask-Login.

Auth is OPTIONAL for using the diagnosis tool itself (guests can still
upload and predict) — logging in just attaches your predictions to your own
account so /history shows only your uploads. Kept intentionally simple:
no email verification, no password reset. Good enough for a small team /
single farm; swap for something stronger before exposing this publicly.
"""

from flask_login import LoginManager, UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

import db

login_manager = LoginManager()
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, row):
        self.id = row["id"]
        self.username = row["username"]
        self.password_hash = row["password_hash"]


@login_manager.user_loader
def load_user(user_id):
    row = db.get_user_by_id(int(user_id))
    return User(row) if row else None


def register_user(username, password):
    if db.get_user_by_username(username):
        return None, "Username already taken."
    if len(password) < 6:
        return None, "Password must be at least 6 characters."
    user_id = db.create_user(username, generate_password_hash(password))
    return db.get_user_by_id(user_id), None


def verify_login(username, password):
    row = db.get_user_by_username(username)
    if row and check_password_hash(row["password_hash"], password):
        return User(row)
    return None
