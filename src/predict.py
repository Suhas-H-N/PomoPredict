"""
Run inference on a single pomegranate leaf/fruit image using the trained
model.

Usage:
    python src/predict.py --image path/to/leaf_or_fruit.jpg
"""

import argparse
import json
import os

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

from model import IMG_SIZE

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "pomegranate_disease_model.h5")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

# Optional: short human-readable notes shown alongside predictions in the app
DISEASE_INFO = {
    "Healthy": "No visible disease symptoms.",
    "Bacterial_Blight": "Xanthomonas axonopodis pv. punicae — dark oily spots, cracking.",
    "Anthracnose": "Colletotrichum gloeosporioides — sunken dark lesions.",
    "Cercospora_Leaf_Spot": "Cercospora spp. — small grey-brown circular leaf spots.",
    "Fruit_Rot": "Alternaria / Aspergillus spp. — soft rot with fungal growth.",
}


def load_class_names():
    with open(CLASS_NAMES_PATH) as f:
        return json.load(f)


def predict_image(img_path, model=None, class_names=None, top_k=3):
    """Returns a list of (class_name, confidence) tuples, sorted descending."""
    if model is None:
        model = load_model(MODEL_PATH)
    if class_names is None:
        class_names = load_class_names()

    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # add batch dimension (model rescales internally)

    preds = model.predict(arr, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]

    return [(class_names[i], float(preds[i])) for i in top_indices]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict pomegranate disease from an image")
    parser.add_argument("--image", type=str, required=True, help="Path to a leaf/fruit image")
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run src/train.py first."
        )

    results = predict_image(args.image, top_k=args.top_k)

    print(f"\nPredictions for: {args.image}\n" + "-" * 45)
    for label, confidence in results:
        note = DISEASE_INFO.get(label, "")
        print(f"{label:<25} {confidence * 100:5.1f}%   {note}")


if __name__ == "__main__":
    main()
