# ============================================
# Brain Tumor Segmentation Flask App
# Using U-Net + VGG16 Transfer Learning (PyTorch)
# ============================================

import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import tifffile
from flask import Flask, render_template, request, jsonify
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — required for Flask
import matplotlib.pyplot as plt

# ============================================
# Flask setup
# ============================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================
# Device setup
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)

# ============================================
# Image transform (same as training — no augmentation)
# ============================================

IMAGE_SIZE = 256

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ============================================
# Load U-Net + VGG16 model
# ============================================

model = smp.Unet(
    encoder_name="vgg16",
    encoder_weights=None,   # weights come from checkpoint, not ImageNet
    in_channels=3,
    classes=1,
    activation=None
)

model_path = "../model/brain_tumor_segmentation_best_model.pth"
checkpoint  = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print(f"Segmentation model loaded — best epoch: {checkpoint['epoch']+1}")
print(f"  Val IoU:  {checkpoint['val_iou']:.4f}")
print(f"  Val Dice: {checkpoint['val_dice']:.4f}")

# ============================================
# Prediction function
# ============================================

def predict_segmentation(image_path, threshold=0.5):
    """
    Load an image, run U-Net inference, and return:
    - base64-encoded overlay PNG (original MRI + red tumor mask)
    - base64-encoded probability heatmap PNG
    - tumor_detected: bool
    - tumor_coverage: float (% of pixels)
    - iou_val, dice_val from saved checkpoint
    """

    # ── Load image ────────────────────────────────────────────
    filename = image_path.lower()

    if filename.endswith(".tif") or filename.endswith(".tiff"):
        image_np = tifffile.imread(image_path)
    else:
        image_np = np.array(Image.open(image_path).convert("RGB"))

    # Ensure uint8 RGB
    if image_np.dtype != np.uint8:
        if image_np.max() > 0:
            image_np = (image_np / image_np.max() * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    # ── Apply transform ───────────────────────────────────────
    dummy_mask = np.zeros(image_np.shape[:2], dtype=np.float32)
    augmented  = transform(image=image_np, mask=dummy_mask)
    tensor     = augmented["image"].unsqueeze(0).to(device)

    # ── Run model ─────────────────────────────────────────────
    with torch.no_grad():
        prob_map  = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
        pred_mask = (prob_map > threshold).astype(np.float32)

    # ── Compute stats ─────────────────────────────────────────
    tumor_pixels   = int(pred_mask.sum())
    total_pixels   = pred_mask.size
    tumor_coverage = round(100 * tumor_pixels / total_pixels, 2)
    tumor_detected = tumor_pixels > 0

    # ── Build display image (undo normalization) ───────────────
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_display = (augmented["image"].permute(1, 2, 0).numpy() * std + mean).clip(0, 1)

    # ── Overlay image: red tumor on MRI ───────────────────────
    overlay = img_display.copy()
    overlay[pred_mask > 0] = [1.0, 0.18, 0.18]  # bright red

    # ── Encode overlay as base64 PNG ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#0a0a0a")
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.88, bottom=0.02)

    titles = ["MRI Scan", "Tumor Mask", "Overlay"]
    images = [img_display, pred_mask, overlay]
    cmaps  = [None, "Reds", None]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=11, fontweight="600",
                     fontfamily="monospace", pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#0a0a0a", dpi=130)
    plt.close(fig)
    buf.seek(0)
    overlay_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # ── Probability heatmap ───────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(4, 4), facecolor="#0a0a0a")
    im = ax2.imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    ax2.axis("off")
    ax2.set_title("Confidence Map", color="white", fontsize=11,
                  fontweight="600", fontfamily="monospace", pad=8)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white", labelcolor="white")

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches="tight",
                 facecolor="#0a0a0a", dpi=130)
    plt.close(fig2)
    buf2.seek(0)
    heatmap_b64 = base64.b64encode(buf2.read()).decode("utf-8")

    return {
        "tumor_detected":  tumor_detected,
        "tumor_coverage":  tumor_coverage,
        "tumor_pixels":    tumor_pixels,
        "total_pixels":    total_pixels,
        "overlay_image":   overlay_b64,
        "heatmap_image":   heatmap_b64,
        "val_iou":         round(checkpoint["val_iou"]  * 100, 1),
        "val_dice":        round(checkpoint["val_dice"] * 100, 1),
        "best_epoch":      checkpoint["epoch"] + 1,
    }

# ============================================
# Routes
# ============================================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"})

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        try:
            result = predict_segmentation(file_path)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("index.html")

# ============================================
# Run
# ============================================

if __name__ == "__main__":
    print("\nStarting Brain Tumor Segmentation App...")
    print("Open browser at: http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=True)# ============================================
# Brain Tumor Segmentation Flask App
# Using U-Net + VGG16 Transfer Learning (PyTorch)
# ============================================

import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import tifffile
from flask import Flask, render_template, request, jsonify
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — required for Flask
import matplotlib.pyplot as plt

# ============================================
# Flask setup
# ============================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================
# Device setup
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)

# ============================================
# Image transform (same as training — no augmentation)
# ============================================

IMAGE_SIZE = 256

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ============================================
# Load U-Net + VGG16 model
# ============================================

model = smp.Unet(
    encoder_name="vgg16",
    encoder_weights=None,   # weights come from checkpoint, not ImageNet
    in_channels=3,
    classes=1,
    activation=None
)

model_path = "brain_tumor_segmentation_best_model.pth"
checkpoint  = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print(f"Segmentation model loaded — best epoch: {checkpoint['epoch']+1}")
print(f"  Val IoU:  {checkpoint['val_iou']:.4f}")
print(f"  Val Dice: {checkpoint['val_dice']:.4f}")

# ============================================
# Prediction function
# ============================================

def predict_segmentation(image_path, threshold=0.5):
    """
    Load an image, run U-Net inference, and return:
    - base64-encoded overlay PNG (original MRI + red tumor mask)
    - base64-encoded probability heatmap PNG
    - tumor_detected: bool
    - tumor_coverage: float (% of pixels)
    - iou_val, dice_val from saved checkpoint
    """

    # ── Load image ────────────────────────────────────────────
    filename = image_path.lower()

    if filename.endswith(".tif") or filename.endswith(".tiff"):
        image_np = tifffile.imread(image_path)
    else:
        image_np = np.array(Image.open(image_path).convert("RGB"))

    # Ensure uint8 RGB
    if image_np.dtype != np.uint8:
        if image_np.max() > 0:
            image_np = (image_np / image_np.max() * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    # ── Apply transform ───────────────────────────────────────
    dummy_mask = np.zeros(image_np.shape[:2], dtype=np.float32)
    augmented  = transform(image=image_np, mask=dummy_mask)
    tensor     = augmented["image"].unsqueeze(0).to(device)

    # ── Run model ─────────────────────────────────────────────
    with torch.no_grad():
        prob_map  = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
        pred_mask = (prob_map > threshold).astype(np.float32)

    # ── Compute stats ─────────────────────────────────────────
    tumor_pixels   = int(pred_mask.sum())
    total_pixels   = pred_mask.size
    tumor_coverage = round(100 * tumor_pixels / total_pixels, 2)
    tumor_detected = tumor_pixels > 0

    # ── Build display image (undo normalization) ───────────────
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_display = (augmented["image"].permute(1, 2, 0).numpy() * std + mean).clip(0, 1)

    # ── Overlay image: red tumor on MRI ───────────────────────
    overlay = img_display.copy()
    overlay[pred_mask > 0] = [1.0, 0.18, 0.18]  # bright red

    # ── Encode overlay as base64 PNG ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#0a0a0a")
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.88, bottom=0.02)

    titles = ["MRI Scan", "Tumor Mask", "Overlay"]
    images = [img_display, pred_mask, overlay]
    cmaps  = [None, "Reds", None]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=11, fontweight="600",
                     fontfamily="monospace", pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#0a0a0a", dpi=130)
    plt.close(fig)
    buf.seek(0)
    overlay_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # ── Probability heatmap ───────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(4, 4), facecolor="#0a0a0a")
    im = ax2.imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    ax2.axis("off")
    ax2.set_title("Confidence Map", color="white", fontsize=11,
                  fontweight="600", fontfamily="monospace", pad=8)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white", labelcolor="white")

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches="tight",
                 facecolor="#0a0a0a", dpi=130)
    plt.close(fig2)
    buf2.seek(0)
    heatmap_b64 = base64.b64encode(buf2.read()).decode("utf-8")

    return {
        "tumor_detected":  tumor_detected,
        "tumor_coverage":  tumor_coverage,
        "tumor_pixels":    tumor_pixels,
        "total_pixels":    total_pixels,
        "overlay_image":   overlay_b64,
        "heatmap_image":   heatmap_b64,
        "val_iou":         round(checkpoint["val_iou"]  * 100, 1),
        "val_dice":        round(checkpoint["val_dice"] * 100, 1),
        "best_epoch":      checkpoint["epoch"] + 1,
    }

# ============================================
# Routes
# ============================================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"})

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        try:
            result = predict_segmentation(file_path)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("index.html")

# ============================================
# Run
# ============================================

if __name__ == "__main__":
    print("\nStarting Brain Tumor Segmentation App...")
    print("Open browser at: http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=True)