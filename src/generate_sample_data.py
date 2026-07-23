"""
Generates a tiny synthetic image dataset so you can smoke-test the full
training/prediction pipeline without needing to download a real dataset
first.

NOTE: These are randomly colored/textured images, NOT real pomegranate
photos, so the trained model will not have any real-world predictive value.
Replace data/sample_dataset with a real dataset for a model that's actually
useful — search Kaggle/Mendeley Data for "Pomegranate Disease Dataset".

Usage:
    python src/generate_sample_data.py
"""

import os
import random

import numpy as np
from PIL import Image, ImageDraw

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_dataset")

# Class name -> (base_color, lesion_color or None)
# Colors loosely mimic the real symptom palette of each disease.
CLASSES = {
    "Healthy": ((60, 150, 60), None),                     # green leaf/fruit
    "Bacterial_Blight": ((120, 60, 40), (30, 20, 15)),     # oily dark blotches
    "Anthracnose": ((140, 90, 50), (50, 30, 20)),          # sunken brown lesions
    "Cercospora_Leaf_Spot": ((70, 130, 60), (110, 100, 80)),  # grey-brown spots on green
    "Fruit_Rot": ((150, 70, 40), (90, 90, 60)),            # rot with fungal growth
}

IMAGES_PER_CLASS = 40
IMG_SIZE = (128, 128)


def make_image(base_color, lesion_color):
    img = Image.new("RGB", IMG_SIZE, base_color)
    draw = ImageDraw.Draw(img)

    # add some texture/noise so the CNN has something to learn
    for _ in range(30):
        x, y = random.randint(0, IMG_SIZE[0]), random.randint(0, IMG_SIZE[1])
        r = random.randint(2, 6)
        jitter = tuple(
            max(0, min(255, c + random.randint(-25, 25))) for c in base_color
        )
        draw.ellipse([x - r, y - r, x + r, y + r], fill=jitter)

    if lesion_color is not None:
        # simulate disease lesions/spots with irregular blotches
        for _ in range(random.randint(5, 14)):
            x, y = random.randint(0, IMG_SIZE[0]), random.randint(0, IMG_SIZE[1])
            r = random.randint(3, 10)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=lesion_color)

    return img


def main():
    random.seed(42)
    np.random.seed(42)

    for class_name, (base_color, lesion_color) in CLASSES.items():
        class_dir = os.path.join(OUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(IMAGES_PER_CLASS):
            img = make_image(base_color, lesion_color)
            img.save(os.path.join(class_dir, f"{class_name}_{i:03d}.jpg"), quality=90)

        print(f"Generated {IMAGES_PER_CLASS} images for {class_name}")

    print(f"\nDone. Synthetic dataset written to: {os.path.abspath(OUT_DIR)}")
    print("Remember: these are synthetic images for pipeline testing only.")


if __name__ == "__main__":
    main()
