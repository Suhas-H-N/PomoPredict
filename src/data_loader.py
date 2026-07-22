"""
Data loading utilities: builds train/validation tf.data.Dataset objects from
a directory of class-labeled sub-folders (see README for expected layout).
"""

import tensorflow as tf

from model import IMG_SIZE


def load_datasets(data_dir, img_size=IMG_SIZE, batch_size=32, val_split=0.2, seed=42):
    """Load train/val datasets and return (train_ds, val_ds, class_names)."""

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    # Light augmentation applied only to the training set to reduce overfitting
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names
