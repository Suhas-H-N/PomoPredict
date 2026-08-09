"""
Tracks trained model versions so you can train multiple times, compare them,
and choose which one the app actually serves — without overwriting old runs.

Layout on disk:
    models/
      registry.json                 <- {"active": "<version_id>", "versions": {...}}
      <version_id>/
        model.h5
        class_names.json
        metrics.json
        training_history.png
"""

import json
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")


def _load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return {"active": None, "versions": {}}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _save_registry(registry):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def register_version(version_id, model_type, val_accuracy, val_loss, num_classes,
                      set_active=True):
    """Add a newly trained version to the registry."""
    registry = _load_registry()
    registry["versions"][version_id] = {
        "model_type": model_type,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
        "num_classes": num_classes,
        "path": os.path.join(MODELS_DIR, version_id),
    }
    if set_active or registry.get("active") is None:
        registry["active"] = version_id
    _save_registry(registry)
    return registry


def list_versions():
    return _load_registry()["versions"]


def get_active_version():
    registry = _load_registry()
    active = registry.get("active")
    if active is None:
        return None
    return active, registry["versions"].get(active)


def set_active_version(version_id):
    registry = _load_registry()
    if version_id not in registry["versions"]:
        raise ValueError(f"Unknown model version: {version_id}")
    registry["active"] = version_id
    _save_registry(registry)


def version_dir(version_id):
    return os.path.join(MODELS_DIR, version_id)
