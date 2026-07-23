"""
CNN architecture for pomegranate disease classification.

The default model is a compact from-scratch CNN that trains quickly on CPU,
which is ideal for learning/demo purposes. A transfer-learning variant
(MobileNetV2) is also provided for higher accuracy on real datasets.
"""

from tensorflow.keras import layers, models, applications

IMG_SIZE = (128, 128)
IMG_CHANNELS = 3


def build_cnn(num_classes: int, img_size=IMG_SIZE):
    """Build a compact CNN from scratch.

    Good for: quick experiments, small datasets, CPU-only training.
    """
    inputs = layers.Input(shape=(img_size[0], img_size[1], IMG_CHANNELS))

    x = layers.Rescaling(1.0 / 255)(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="pomegranate_disease_cnn")
    return model


def build_transfer_model(num_classes: int, img_size=IMG_SIZE):
    """Build a transfer-learning model using MobileNetV2 as a frozen backbone.

    Good for: real datasets, higher accuracy, when you have GPU access or can
    tolerate slower CPU training.
    """
    base = applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], IMG_CHANNELS),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze backbone; unfreeze top layers later for fine-tuning

    inputs = layers.Input(shape=(img_size[0], img_size[1], IMG_CHANNELS))
    x = applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="pomegranate_disease_transfer")
    return model


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    # Keras 3 needs learning_rate set on the optimizer instance, so re-set it
    model.optimizer.learning_rate = learning_rate
    return model
