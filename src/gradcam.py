"""
Grad-CAM: highlights which pixels of the input image most influenced the
model's predicted class, so a user can sanity-check *why* the model called
a leaf diseased instead of just trusting a number.

Also used as a cheap proxy for "severity": the fraction of the leaf/fruit
covered by a strong heat signal correlates with how much of the surface
looks abnormal to the model.
"""

import io

import numpy as np
import tensorflow as tf
from PIL import Image


def _find_last_conv_layer(model):
    """Walk backwards through the model to find the last Conv2D-like layer."""
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:  # (batch, H, W, channels)
            return layer.name
    raise ValueError("Could not find a 4D (conv) layer for Grad-CAM.")


def compute_gradcam(model, img_array, class_index, last_conv_layer_name=None):
    """
    img_array: preprocessed batch of shape (1, H, W, 3) — same array you'd feed
    to model.predict().
    Returns a (H, W) heatmap normalized to [0, 1].
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def heatmap_coverage(heatmap, threshold=0.5):
    """Fraction of the heatmap above `threshold` — used as a severity proxy."""
    return float((heatmap >= threshold).sum()) / float(heatmap.size)


def overlay_heatmap_on_image(pil_image, heatmap, alpha=0.4):
    """Returns a new PIL image with a red/yellow heatmap overlaid."""
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(pil_image.size)
    heatmap_arr = np.array(heatmap_img)

    # simple red-yellow colormap without needing matplotlib's cm module
    colored = np.zeros((*heatmap_arr.shape, 3), dtype=np.uint8)
    colored[..., 0] = 255  # red channel full
    colored[..., 1] = heatmap_arr  # green ramps up with intensity -> yellow
    colored_img = Image.fromarray(colored).convert("RGB")

    base = pil_image.convert("RGB")
    blended = Image.blend(base, colored_img, alpha=alpha)
    return blended


def heatmap_to_data_url(pil_image, heatmap, alpha=0.4):
    """Convenience: overlay + encode as a base64 PNG data URL for embedding in HTML."""
    import base64

    blended = overlay_heatmap_on_image(pil_image, heatmap, alpha=alpha)
    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def severity_label(coverage):
    """Turn a 0-1 heatmap coverage fraction into a human severity bucket."""
    if coverage < 0.10:
        return "Mild"
    if coverage < 0.30:
        return "Moderate"
    return "Severe"
