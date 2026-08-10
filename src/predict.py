"""
Run inference on a pomegranate leaf/fruit image using the active trained
model (see model_registry.py for how "active" is chosen).

Usage:
    python src/predict.py --image path/to/leaf_or_fruit.jpg
    python src/predict.py --image path/to/leaf_or_fruit.jpg --tta --gradcam out.png
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

from gradcam import compute_gradcam, heatmap_coverage, heatmap_to_data_url, severity_label
from model import IMG_SIZE
from model_registry import get_active_version, version_dir

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Fallback paths, used only if no versioned model has been registered yet
# (i.e. an older checkout that trained before model_registry.py existed).
MODEL_PATH = os.path.join(MODELS_DIR, "pomegranate_disease_model.h5")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

# Below this confidence, we report "Uncertain" rather than a specific disease —
# a low-confidence top guess is often noise (bad lighting, non-leaf photo, etc).
UNCERTAIN_THRESHOLD = 0.40
UNKNOWN_LABEL = "Uncertain / Not confidently classified"

# Short human-readable notes + suggested treatment/action, shown in the app.
DISEASE_INFO = {
    "Healthy": {
        "note": "No visible disease symptoms.",
        "treatment": "No action needed — keep up regular watering and monitoring.",
    },
    "Bacterial_Blight": {
        "note": "Xanthomonas axonopodis pv. punicae — dark oily spots, cracking.",
        "treatment": ("Prune and destroy infected twigs/fruit, avoid overhead irrigation, "
                       "and apply a copper-based bactericide (e.g. Bordeaux mixture / "
                       "copper oxychloride) per local agricultural guidelines."),
    },
    "Anthracnose": {
        "note": "Colletotrichum gloeosporioides — sunken dark lesions.",
        "treatment": ("Remove and destroy affected fruit/leaves, improve air circulation "
                       "by pruning, and apply a labeled fungicide (e.g. carbendazim or "
                       "mancozeb) at the recommended interval."),
    },
    "Cercospora_Leaf_Spot": {
        "note": "Cercospora spp. — small grey-brown circular leaf spots.",
        "treatment": ("Remove fallen/infected leaves to reduce spore load, avoid wetting "
                       "foliage when watering, and apply a copper or chlorothalonil-based "
                       "fungicide if spread continues."),
    },
    "Fruit_Rot": {
        "note": "Alternaria / Aspergillus spp. — soft rot with fungal growth.",
        "treatment": ("Remove and dispose of rotten fruit away from the orchard, avoid "
                       "fruit injury during harvest/handling, and apply a preventive "
                       "fungicide spray during the susceptible fruiting stage."),
    },
}


def _resolve_model_paths():
    """Prefer the active registered version; fall back to the legacy flat path."""
    active = get_active_version()
    if active is not None:
        version_id, meta = active
        vdir = version_dir(version_id)
        return (
            os.path.join(vdir, "model.h5"),
            os.path.join(vdir, "class_names.json"),
            version_id,
        )
    return MODEL_PATH, CLASS_NAMES_PATH, "legacy"


def load_class_names(class_names_path=None):
    path = class_names_path or _resolve_model_paths()[1]
    with open(path) as f:
        return json.load(f)


def load_active_model():
    model_path, class_names_path, version_id = _resolve_model_paths()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run src/train.py first."
        )
    model = load_model(model_path)
    class_names = load_class_names(class_names_path)
    return model, class_names, version_id


def _load_and_preprocess(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img)
    return img, arr


def _tta_variants(arr):
    """A handful of cheap test-time-augmentation views: original, h-flip, small rotations."""
    variants = [arr, np.fliplr(arr)]
    pil = Image.fromarray(arr.astype("uint8"))
    for angle in (-8, 8):
        rotated = pil.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        variants.append(keras_image.img_to_array(rotated))
    return variants


def predict_image(img_path, model=None, class_names=None, top_k=3, use_tta=False,
                   gradcam=False):
    """
    Returns a dict:
        {
          "raw": [(label, confidence), ...]          # top_k, sorted desc
          "top_label": str, "top_confidence": float,
          "is_uncertain": bool,
          "severity": str | None,
          "gradcam_data_url": str | None
        }
    """
    version_id = None
    if model is None or class_names is None:
        model, class_names, version_id = load_active_model()

    pil_img, arr = _load_and_preprocess(img_path)

    if use_tta:
        batch = np.stack(_tta_variants(arr), axis=0)
        preds = model.predict(batch, verbose=0)
        preds = preds.mean(axis=0)
    else:
        batch = np.expand_dims(arr, axis=0)
        preds = model.predict(batch, verbose=0)[0]

    top_indices = preds.argsort()[-top_k:][::-1]
    raw = [(class_names[i], float(preds[i])) for i in top_indices]

    top_label, top_confidence = raw[0]
    is_uncertain = top_confidence < UNCERTAIN_THRESHOLD

    severity = None
    gradcam_data_url = None
    if gradcam and not is_uncertain and top_label != "Healthy":
        try:
            single_batch = np.expand_dims(arr, axis=0)
            top_class_index = class_names.index(top_label)
            heatmap = compute_gradcam(model, single_batch, top_class_index)
            coverage = heatmap_coverage(heatmap)
            severity = severity_label(coverage)
            gradcam_data_url = heatmap_to_data_url(pil_img, heatmap)
        except Exception:
            # Grad-CAM is a bonus feature — never let it break a prediction.
            severity = None
            gradcam_data_url = None

    return {
        "raw": raw,
        "top_label": UNKNOWN_LABEL if is_uncertain else top_label,
        "top_confidence": top_confidence,
        "is_uncertain": is_uncertain,
        "severity": severity,
        "gradcam_data_url": gradcam_data_url,
        "model_version": version_id,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Predict pomegranate disease from an image")
    parser.add_argument("--image", type=str, required=True, help="Path to a leaf/fruit image")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--gradcam", type=str, default=None,
                         help="If set, save a Grad-CAM overlay PNG to this path")
    return parser.parse_args()


def main():
    args = parse_args()
    result = predict_image(
        args.image, top_k=args.top_k, use_tta=args.tta, gradcam=bool(args.gradcam)
    )

    print(f"\nPredictions for: {args.image}  (model: {result['model_version']})\n" + "-" * 55)
    for label, confidence in result["raw"]:
        note = DISEASE_INFO.get(label, {}).get("note", "")
        print(f"{label:<25} {confidence * 100:5.1f}%   {note}")

    if result["is_uncertain"]:
        print(f"\n⚠ {UNKNOWN_LABEL} (top confidence {result['top_confidence']*100:.1f}% "
              f"< {UNCERTAIN_THRESHOLD*100:.0f}% threshold)")

    if result["severity"]:
        print(f"Estimated severity: {result['severity']}")

    if args.gradcam and result["gradcam_data_url"]:
        import base64
        header, encoded = result["gradcam_data_url"].split(",", 1)
        with open(args.gradcam, "wb") as f:
            f.write(base64.b64decode(encoded))
        print(f"Saved Grad-CAM overlay to {args.gradcam}")


if __name__ == "__main__":
    main()
