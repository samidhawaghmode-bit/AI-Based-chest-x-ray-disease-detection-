import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -----------------------------
# Folders
# -----------------------------
UPLOAD_FOLDER = 'static/uploads'
HEATMAP_FOLDER = 'static/heatmaps'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = 'covid_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)
model.trainable = False

# -----------------------------
# Classes & image size
# -----------------------------
classes = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
IMG_SIZE = (160, 160)

# -----------------------------
# Helpers
# -----------------------------
def preprocess_image_cv(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_array = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)
    return img_array, img_rgb


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if hasattr(layer, "output_shape"):
            shape = layer.output_shape
            if isinstance(shape, tuple) and len(shape) == 4:
                return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()


def make_saliency_map(img_array, model, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.cast(img_tensor, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]
    grads = tape.gradient(loss, img_tensor)
    grads = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    grads = (grads - tf.reduce_min(grads)) / (tf.reduce_max(grads) - tf.reduce_min(grads) + 1e-8)
    return grads.numpy()


def save_and_overlay_heatmap(heatmap, original_img_rgb, out_heatmap_path, out_overlay_path, alpha=0.4):
    h_orig, w_orig = original_img_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w_orig, h_orig))
    hm_uint8 = np.uint8(255 * heatmap_resized)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    orig_bgr = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(hm_color, alpha, orig_bgr, 1 - alpha, 0)
    cv2.imwrite(out_heatmap_path, hm_color)
    cv2.imwrite(out_overlay_path, overlay)
    return out_overlay_path, out_heatmap_path


def bbox_from_heatmap(heatmap, original_img_rgb, threshold=0.6, min_area=500):
    h_orig, w_orig = original_img_rgb.shape[:2]
    hm_resized = cv2.resize(heatmap, (w_orig, h_orig))
    mask = np.uint8((hm_resized >= threshold) * 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            boxes.append((x, y, w, h))
    return boxes


# -----------------------------
# Severity Estimation
# -----------------------------
def estimate_severity_from_heatmap(heatmap, threshold=0.5):
    """
    Estimate severity based on percentage of active pixels in the heatmap.
    """
    active_pixels = np.sum(heatmap >= threshold)
    total_pixels = heatmap.size
    ratio = active_pixels / total_pixels

    if ratio < 0.10:
        return "Mild"
    elif ratio < 0.25:
        return "Moderate"
    else:
        return "Severe"


# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('home.html')


@app.route("/suggestions")
def suggestions():
    return render_template("s.html")


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'xrayFile' not in request.files:
        return redirect(request.url)
    file = request.files['xrayFile']
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img_array, original_img = preprocess_image_cv(filepath)
    if img_array is None:
        return "❌ Could not read uploaded image."

    preds = model.predict(img_array)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][class_idx]) * 100.0
    pred_class = classes[class_idx].upper()

    # If NORMAL → no heatmap needed
    if pred_class == "NORMAL":
        return render_template(
            'result.html',
            original_img=filepath.replace('\\', '/'),
            disease=pred_class,
            confidence=round(confidence, 2),
            severity="None",
            explain_method="Not applicable",
            is_normal=True
        )

    # Otherwise abnormal → generate XAI
    last_conv = find_last_conv_layer(model)
    if last_conv:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv, pred_index=class_idx)
        explain_method = "Grad-CAM"
    else:
        heatmap = make_saliency_map(img_array, model, pred_index=class_idx)
        explain_method = "Saliency Map (fallback)"

    base_name = os.path.splitext(filename)[0]
    heatmap_path = os.path.join(HEATMAP_FOLDER, f"{base_name}_heatmap.png")
    overlay_path = os.path.join(HEATMAP_FOLDER, f"{base_name}_overlay.png")
    overlay_saved, heatmap_saved = save_and_overlay_heatmap(heatmap, original_img, heatmap_path, overlay_path)

    boxes = bbox_from_heatmap(heatmap, original_img)
    boxed_path = None
    if boxes:
        boxed_img = original_img.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(boxed_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        boxed_bgr = cv2.cvtColor(boxed_img, cv2.COLOR_RGB2BGR)
        boxed_path = os.path.join(HEATMAP_FOLDER, f"{base_name}_boxes.png")
        cv2.imwrite(boxed_path, boxed_bgr)

    # ✅ Compute severity realistically
    severity = estimate_severity_from_heatmap(heatmap)

    return render_template(
        'result.html',
        original_img=filepath.replace('\\', '/'),
        xai_overlay=overlay_saved.replace('\\', '/'),
        xai_heatmap=heatmap_saved.replace('\\', '/'),
        boxed_img=boxed_path.replace('\\', '/') if boxed_path else None,
        disease=pred_class,
        confidence=round(confidence, 2),
        severity=severity,
        explain_method=explain_method,
        is_normal=False
    )


if __name__ == '__main__':
    app.run(debug=True)
