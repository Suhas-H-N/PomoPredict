# PomoPredict 🍎

A Flask + Keras app that diagnoses pomegranate leaf/fruit diseases from a
photo — upload, drag-drop batch, or live webcam capture.

## Features

- **CNN or MobileNetV2 transfer-learning model**, trained on your own labeled
  image folders (`src/train.py`)
- **Batch upload** — diagnose several photos in one request
- **Webcam capture** — point a camera and diagnose a live frame
- **Grad-CAM heatmaps** — see *which pixels* drove the prediction, not just a
  confidence number
- **Severity estimate** (Mild/Moderate/Severe), derived from Grad-CAM
  heatmap coverage
- **Uncertainty handling** — low-confidence predictions are reported as
  "Uncertain" instead of a false-confident guess
- **Test-time augmentation** — averages predictions over flipped/rotated
  views of the input for a small accuracy boost
- **Treatment notes** per disease class
- **Optional accounts** (Flask-Login) — logged-in users get:
  - **History** of past predictions, stored in SQLite
  - **Geotagged map** of predictions (if you allow location access)
  - **Downloadable PDF report** per prediction
  - **"This is wrong" correction flow** — flags a prediction and copies the
    image into `data/corrections/<label>/` for your next retrain
- **Model registry** — every training run is saved as its own version; pick
  which one the app serves from `/admin/models` (admin-only)
- **JSON REST API** at `/api/predict` for other clients (scripts, mobile apps)
- **Dockerfile** for deployment behind gunicorn

## Quickstart

```bash
pip install -r requirements.txt

# Optional: generate a small synthetic dataset just to smoke-test the pipeline
# (these are NOT real pomegranate photos — replace with a real dataset before
# trusting predictions; e.g. search Kaggle/Mendeley for "Pomegranate Disease")
python src/generate_sample_data.py

python src/train.py --data_dir data/sample_dataset --epochs 15 --model_type cnn

python app.py
# open http://127.0.0.1:5000
```

## Expected data layout for training

```
data/sample_dataset/
  Healthy/            *.jpg
  Bacterial_Blight/    *.jpg
  Anthracnose/         *.jpg
  Cercospora_Leaf_Spot/*.jpg
  Fruit_Rot/           *.jpg
```

Any set of class-named sub-folders works — swap in your own dataset with
different classes and `--data_dir` and everything downstream (labels,
treatment notes fallback, etc.) still runs; just add matching entries to
`DISEASE_INFO` in `src/predict.py` if you add new classes.

## Model versions

Every `src/train.py` run is saved under `models/<version_id>/` and
registered in `models/registry.json` rather than overwriting the previous
model. Visit `/admin/models` (while logged in as an admin) to see all
trained versions and switch which one is active without retraining.

Admin access defaults to the **first registered account** (user id 1). To
name specific admins instead, set:

```bash
export ADMIN_USERNAMES="alice,bob"
```

## Configuration

| Env var          | Default               | Purpose                                   |
|-------------------|------------------------|--------------------------------------------|
| `SECRET_KEY`      | `dev-only-change-me`  | Flask session signing key — **set this** before deploying anywhere reachable by others |
| `ADMIN_USERNAMES` | *(unset)*              | Comma-separated usernames allowed to change the active model |
| `FLASK_DEBUG`     | `0`                    | Set to `1` for the Werkzeug debugger locally. **Never enable in production** — it allows remote code execution. |

## REST API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "image=@leaf.jpg" \
  -F "lat=12.97" -F "lon=77.59"
```

Returns:

```json
{
  "raw": [["Bacterial_Blight", 0.82], ["Anthracnose", 0.11], ["Healthy", 0.04]],
  "top_label": "Bacterial_Blight",
  "top_confidence": 0.82,
  "is_uncertain": false,
  "severity": "Moderate",
  "gradcam_data_url": "data:image/png;base64,...",
  "treatment": "Prune and destroy infected twigs/fruit...",
  "prediction_id": 14
}
```

`GET /api/history` (requires login) returns the calling user's saved
predictions as JSON.

## Docker

```bash
docker build -t pomopredict .
docker run -p 5000:5000 -e SECRET_KEY=$(openssl rand -hex 32) pomopredict
```

The image runs a single gunicorn worker — the trained model is loaded lazily
into a process-global, which isn't currently safe to share across multiple
worker processes.

## Known limitations / next steps

- **Not production-hardened**: there's no CSRF protection on state-changing
  forms (login, corrections, admin) — fine for a small trusted team behind a
  private network, not for a public-facing deployment. Add
  [Flask-WTF](https://flask-wtf.readthedocs.io/) CSRF tokens before exposing
  this publicly.
- No rate limiting on `/api/predict` — add one (e.g. Flask-Limiter) if this
  is public-facing.
- `src/generate_sample_data.py` produces synthetic colored blobs, not real
  disease photos — trained models from it have no real-world accuracy. Swap
  in a real labeled dataset before relying on predictions.
- Corrections collected via the "flag wrong" flow are copied to
  `data/corrections/<label>/` but not automatically merged into the next
  training run — copy them into your `data_dir` yourself before retraining.

Educational demo — not a substitute for advice from a certified agronomist.
