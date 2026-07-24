"""
Train the pomegranate disease classifier.

Usage:
    python src/train.py --data_dir data/sample_dataset --epochs 15 --model_type cnn
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from data_loader import load_datasets
from model import IMG_SIZE, build_cnn, build_transfer_model, compile_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def parse_args():
    parser = argparse.ArgumentParser(description="Train pomegranate disease CNN")
    parser.add_argument("--data_dir", type=str, default="data/sample_dataset",
                         help="Path to folder containing one sub-folder per class")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--model_type", choices=["cnn", "transfer"], default="cnn",
                         help="'cnn' = fast from-scratch model, "
                              "'transfer' = MobileNetV2 backbone (higher accuracy)")
    return parser.parse_args()


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    args = parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Loading data from {args.data_dir} ...")
    train_ds, val_ds, class_names = load_datasets(
        args.data_dir, img_size=IMG_SIZE, batch_size=args.batch_size
    )
    print(f"Found {len(class_names)} classes: {class_names}")

    if args.model_type == "transfer":
        model = build_transfer_model(len(class_names), img_size=IMG_SIZE)
    else:
        model = build_cnn(len(class_names), img_size=IMG_SIZE)

    model = compile_model(model, learning_rate=args.learning_rate)
    model.summary()

    import tensorflow as tf
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model_path = os.path.join(MODELS_DIR, "pomegranate_disease_model.h5")
    model.save(model_path)
    print(f"Saved trained model to {model_path}")

    class_names_path = os.path.join(MODELS_DIR, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved class names to {class_names_path}")

    plot_history(history, os.path.join(MODELS_DIR, "training_history.png"))

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\nFinal validation accuracy: {val_acc:.4f} | loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
