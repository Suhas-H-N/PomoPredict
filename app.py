"""
Flask web app: upload a pomegranate leaf/fruit image, get back the predicted
disease with a confidence score.

Usage:
    python app.py
    then open http://127.0.0.1:5000
"""

import os
import sys

from flask import Flask, render_template, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from predict import DISEASE_INFO, MODEL_PATH, load_class_names, predict_image  # noqa: E402

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

_model = None
_class_names = None


def get_model():
    """Lazy-load the model so the app starts instantly even before training."""
    global _model, _class_names
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
        _class_names = load_class_names()
    return _model, _class_names


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    image_url = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            image_url = filepath

            if not os.path.exists(MODEL_PATH):
                error = ("No trained model found. Run 'python src/train.py' "
                          "first to train a model.")
            else:
                model, class_names = get_model()
                raw = predict_image(filepath, model=model, class_names=class_names)
                predictions = [
                    (label, confidence, DISEASE_INFO.get(label, ""))
                    for label, confidence in raw
                ]
        else:
            error = "Please choose an image to upload."

    return render_template(
        "index.html", predictions=predictions, image_url=image_url, error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
