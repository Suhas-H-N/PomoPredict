"""
Flask web app: upload (or webcam-capture) a pomegranate leaf/fruit image,
get back the predicted disease, a confidence score, an estimated severity,
a Grad-CAM "why" heatmap, and a suggested treatment.

Also provides:
  - batch upload (multiple images at once)
  - optional login so prediction history is tied to your account
  - a map of geotagged predictions
  - a JSON REST API for other clients (mobile apps, scripts)
  - downloadable PDF reports
  - a "this is wrong" correction flow that feeds src/train.py retraining
  - an admin page to switch which trained model version is active

Usage:
    python app.py
    then open http://127.0.0.1:5000
"""

import os
import sys
import uuid

from flask import (Flask, jsonify, redirect, render_template, request,
                    send_file, url_for)
from flask_login import (current_user, login_required, login_user,
                          logout_user)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import auth  # noqa: E402
import db  # noqa: E402
from model_registry import list_versions, set_active_version  # noqa: E402
from predict import DISEASE_INFO, load_active_model, predict_image  # noqa: E402
from report import build_pdf_report  # noqa: E402

app = Flask(__name__, template_folder="templetes")
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-change-me")
# Caps a single request body (guards against someone hammering /api/predict
# or the upload form with huge files to exhaust disk/memory).
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

if app.config["SECRET_KEY"] == "dev-only-change-me" and not app.debug:
    import warnings
    warnings.warn(
        "Using the default SECRET_KEY outside of debug mode. Set the "
        "SECRET_KEY environment variable before deploying this anywhere "
        "reachable by others.", stacklevel=1,
    )

auth.login_manager.init_app(app)
db.init_db()

# Usernames allowed to manage which trained model version is active
# (comma-separated). If unset, only the first registered account (id 1) can.
ADMIN_USERNAMES = {
    u.strip() for u in os.environ.get("ADMIN_USERNAMES", "").split(",") if u.strip()
}

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

_model = None
_class_names = None
_model_version = None


def get_model():
    """Lazy-load (and cache) the currently active model so the app starts instantly."""
    global _model, _class_names, _model_version
    if _model is None:
        _model, _class_names, _model_version = load_active_model()
    return _model, _class_names, _model_version


def reload_model():
    global _model, _class_names, _model_version
    _model = None
    return get_model()


def active_version_id():
    return _model_version


def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _is_admin(user):
    if not user.is_authenticated:
        return False
    if ADMIN_USERNAMES:
        return user.username in ADMIN_USERNAMES
    return user.id == 1  # first account created, if no explicit admins configured


def _owns_prediction(pred, user):
    """Guests (user_id NULL) can't be matched back to anyone, so only the
    logged-in owner (or an admin) may access their saved predictions."""
    if pred.get("user_id") is None:
        return False
    return user.is_authenticated and (pred["user_id"] == user.id or _is_admin(user))


def _save_upload(file_storage):
    ext = file_storage.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_storage.save(filepath)
    return filepath


def _run_prediction(filepath, lat=None, lon=None):
    """Shared by the web form, webcam capture, and the JSON API."""
    model, class_names, version_id = get_model()
    result = predict_image(filepath, model=model, class_names=class_names,
                            use_tta=True, gradcam=True)

    pred_id = db.add_prediction(
        user_id=current_user.id if current_user.is_authenticated else None,
        image_path=filepath,
        top_label=result["top_label"],
        top_confidence=result["top_confidence"],
        severity=result["severity"],
        model_version=version_id,
        all_predictions=result["raw"],
        lat=lat,
        lon=lon,
    )
    result["prediction_id"] = pred_id
    result["treatment"] = DISEASE_INFO.get(result["top_label"], {}).get("treatment")
    result["note"] = DISEASE_INFO.get(result["top_label"], {}).get("note")
    return result


# ------------------------------------------------------------------ pages --
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error = None

    if request.method == "POST":
        files = [f for f in request.files.getlist("image") if f and f.filename]
        lat = request.form.get("lat") or None
        lon = request.form.get("lon") or None

        if not files:
            error = "Please choose at least one image to upload."
        else:
            try:
                get_model()
            except FileNotFoundError:
                error = ("No trained model found. Run 'python src/train.py' "
                          "first to train a model.")

        if not error:
            for f in files:
                if not _allowed(f.filename):
                    error = f"Unsupported file type: {f.filename}"
                    break
                filepath = _save_upload(f)
                r = _run_prediction(filepath, lat=lat, lon=lon)
                r["image_url"] = filepath.replace(os.sep, "/")
                results.append(r)

    return render_template("index.html", results=results, error=error,
                            disease_info=DISEASE_INFO)


@app.route("/webcam", methods=["GET"])
def webcam():
    return render_template("webcam.html")


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """Used by the webcam-capture page: same as /api/predict but redirects to a result view."""
    file = request.files.get("image")
    if not file or not file.filename or not _allowed(file.filename):
        return jsonify({"error": "Please attach a valid image file (png/jpg/jpeg/webp)."}), 400
    filepath = _save_upload(file)
    try:
        result = _run_prediction(filepath, lat=request.form.get("lat"),
                                  lon=request.form.get("lon"))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(result)


# -------------------------------------------------------------------- api --
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """POST multipart/form-data with an 'image' field. Returns JSON."""
    file = request.files.get("image")
    if not file or not file.filename or not _allowed(file.filename):
        return jsonify({"error": "Please attach a valid image file (png/jpg/jpeg/webp)."}), 400

    try:
        get_model()
    except FileNotFoundError:
        return jsonify({"error": "No trained model found on the server."}), 503

    filepath = _save_upload(file)
    lat = request.form.get("lat") or request.args.get("lat")
    lon = request.form.get("lon") or request.args.get("lon")
    result = _run_prediction(filepath, lat=lat, lon=lon)
    result["image_url"] = filepath.replace(os.sep, "/")
    return jsonify(result)


@app.route("/api/history")
@login_required
def api_history():
    rows = db.list_predictions(user_id=current_user.id)
    return jsonify(rows)


# ------------------------------------------------------------------- auth --
@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user_row, error = auth.register_user(username, password)
        if user_row:
            login_user(auth.User(user_row))
            return redirect(url_for("index"))
    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = auth.verify_login(username, password)
        if user:
            login_user(user)
            return redirect(url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


# ---------------------------------------------------------------- history --
@app.route("/history")
@login_required
def history():
    rows = db.list_predictions(user_id=current_user.id)
    geotagged = [r for r in rows if r["lat"] and r["lon"]]
    return render_template("history.html", predictions=rows, geotagged=geotagged)


@app.route("/correct/<int:prediction_id>", methods=["POST"])
@login_required
def correct(prediction_id):
    """Active-learning loop: user flags a wrong prediction with the right label.
    The image is copied into data/corrections/<label>/ for a future retrain."""
    correct_label = request.form.get("correct_label")
    pred = db.get_prediction(prediction_id)
    if not pred or not correct_label:
        return redirect(url_for("history"))
    if not _owns_prediction(pred, current_user):
        return "Not authorized", 403
    if correct_label not in DISEASE_INFO:
        return "Unknown label", 400

    db.set_correction(prediction_id, correct_label)

    import shutil
    corrections_dir = os.path.join("data", "corrections", correct_label)
    os.makedirs(corrections_dir, exist_ok=True)
    if os.path.exists(pred["image_path"]):
        dest = os.path.join(corrections_dir, os.path.basename(pred["image_path"]))
        shutil.copy(pred["image_path"], dest)

    return redirect(url_for("history"))


@app.route("/report/<int:prediction_id>")
@login_required
def report(prediction_id):
    pred = db.get_prediction(prediction_id)
    if not pred:
        return "Not found", 404
    if not _owns_prediction(pred, current_user):
        return "Not authorized", 403
    pdf_path = build_pdf_report(pred, DISEASE_INFO)
    return send_file(pdf_path, as_attachment=True,
                      download_name=f"pomopredict_report_{prediction_id}.pdf")


# ----------------------------------------------------------------- admin --
@app.route("/admin/models", methods=["GET", "POST"])
@login_required
def admin_models():
    if not _is_admin(current_user):
        return "Not authorized", 403

    if request.method == "POST":
        version_id = request.form.get("version_id")
        versions = list_versions()
        if version_id and version_id in versions:
            set_active_version(version_id)
            reload_model()
        return redirect(url_for("admin_models"))

    versions = list_versions()
    return render_template("admin_models.html", versions=versions,
                            active=active_version_id())


# -------------------------------------------------------------- errors --
@app.errorhandler(413)
def too_large(_e):
    if request.path.startswith("/api/") or request.path == "/predict_frame":
        return jsonify({"error": "File too large (16 MB max)."}), 413
    return render_template("index.html", results=[], disease_info=DISEASE_INFO,
                            error="That file is too large — 16 MB max."), 413


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")
