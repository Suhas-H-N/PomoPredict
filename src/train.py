"""
Train the pomegranate disease classifier.

Every run is saved as its own version under models/<version_id>/ and
registered in models/registry.json, so old models are never overwritten and
you can switch which one the app serves (see model_registry.py / app.py's
/admin/models page).

Usage:
    python src/train.py --data_dir data/sample_dataset --epochs 15 --model_type cnn
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from data_loader import load_datasets
from model import IMG_SIZE, build_cnn, build_transfer_model, compile_model
from model_registry import register_version, version_dir

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
    parser.add_argument("--no_activate", action="store_true",
                         help="Train and save this version WITHOUT making it the active model")
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

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\nFinal validation accuracy: {val_acc:.4f} | loss: {val_loss:.4f}")

    # --- Save this run as its own versioned folder --------------------------
    version_id = f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = version_dir(version_id)
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "model.h5")
    model.save(model_path)
    print(f"Saved trained model to {model_path}")

    class_names_path = os.path.join(out_dir, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)

    metrics = {
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "epochs_trained": len(history.history["loss"]),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_history(history, os.path.join(out_dir, "training_history.png"))

    register_version(
        version_id,
        model_type=args.model_type,
        val_accuracy=float(val_acc),
        val_loss=float(val_loss),
        num_classes=len(class_names),
        set_active=not args.no_activate,
    )
    print(f"\nRegistered model version '{version_id}'"
          f"{' and set as active' if not args.no_activate else ''}.")
    print("Manage versions any time in the app under /admin/models.")


if __name__ == "__main__":
    main()
