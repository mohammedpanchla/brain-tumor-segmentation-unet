# 🧠 Brain Tumor Segmentation System
### Pixel-Level Tumor Detection using U-Net + VGG16 Transfer Learning (PyTorch)

---

## 📌 Project Overview

Brain tumors affect approximately **308,000 people worldwide every year**. Accurate localization of the tumor region is critical for surgical planning, treatment decisions, and patient outcomes.

This project upgrades a classification system (V1.0) to a **pixel-level segmentation system (V2.0)** that identifies the exact tumor boundary in brain MRI scans — not just whether a tumor exists, but precisely **where it is**.

The system is built using **U-Net with VGG16 encoder** and deployed as a **Flask web application** with a clinical-grade interface.

### LINK: https://muhammedpanchla-neuroscan-ai.hf.space/#

You can try the model using the MRI images provided in the test_samples/ folder. Simply upload them to the web application (link provided above) or run the Flask app locally to see real tumor segmentation results.

---

## 🎯 Project Objective

The primary goal is to produce a **pixel-wise binary mask** for each MRI scan:

```
Input:  Brain MRI scan (256 × 256)
Output: Binary segmentation mask (256 × 256)
```

Pixel interpretation:

| Value | Meaning |
|---|---|
| 1 | Tumor region |
| 0 | Healthy tissue / background |

This enables clinical-grade tumor localization suitable for radiologist decision support.

---

## 🔬 From V1.0 Classification → V2.0 Segmentation

This project is a direct upgrade from a previous classification model:

| Aspect | V1.0 Classification | V2.0 Segmentation |
|---|---|---|
| Task | One label per image | One label per pixel |
| Output | "Glioma / Meningioma / ..." | 256×256 tumor mask |
| Architecture | VGG16 + classifier head | U-Net + VGG16 encoder |
| Loss Function | CrossEntropyLoss | BCE + Dice Loss |
| Metric | Accuracy | IoU + Dice Score |
| Clinical Value | Tumor type screening | Surgical boundary planning |

---

## 🧠 Architecture: U-Net with VGG16 Encoder

U-Net is the gold standard architecture for medical image segmentation.

```
Input MRI (256×256×3)
│
├── VGG16 Encoder Block 1 → 256×256×64  ──────────────────┐ skip
│         ↓ MaxPool                                        │
├── VGG16 Encoder Block 2 → 128×128×128 ─────────────────┐│ skip
│         ↓ MaxPool                                       ││
├── VGG16 Encoder Block 3 → 64×64×256   ────────────────┐ ││ skip
│         ↓ MaxPool                                      │ ││
├── VGG16 Encoder Block 4 → 32×32×512   ───────────────┐ │ ││ skip
│         ↓ MaxPool                                     │ │ ││
│     BOTTLENECK  16×16×512                             │ │ ││
│         ↑ Upsample                                    │ │ ││
├── Decoder Block 4 ←──────────────────────────────────┘ │ ││
│         ↑ Upsample                                      │ ││
├── Decoder Block 3 ←────────────────────────────────────┘ ││
│         ↑ Upsample                                        ││
├── Decoder Block 2 ←──────────────────────────────────────┘│
│         ↑ Upsample                                         │
├── Decoder Block 1 ←───────────────────────────────────────┘
│
Output Mask (256×256×1) — tumor probability per pixel
```

**Transfer Learning Strategy:**
- **Encoder (VGG16):** Pretrained on ImageNet — carries over from V1.0
- **Decoder:** Trained from scratch on brain MRI segmentation data
- **Benefit:** Encoder already understands edges, textures, and shapes from 1.2M images

---

# 🧠 Trained Model

The trained brain tumor segmentation model is publicly available on Hugging Face.

Hugging Face Repository:
https://huggingface.co/muhammedpanchla/brain-tumor-segmentation-unet-vgg16/tree/main

You can download the final trained model checkpoint file:

brain_tumor_segmentation_best_model.pth

After downloading, place the file inside the following directory:

```
model/
└── brain_tumor_segmentation_best_model.pth
```

This file contains the fully trained U-Net model with VGG16 encoder and can be used directly for:

* Inference
* Deployment
* Further fine-tuning
* Research and experimentation

This ensures reproducibility and allows anyone to use the trained segmentation model without retraining.

---
## 📊 Dataset

**Dataset:** LGG Brain MRI Segmentation — [kaggle_3m](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
**Source:** The Cancer Genome Atlas (TCGA)
**Patients:** 110 patients diagnosed with Lower Grade Glioma (LGG)
**Total pairs:** 3,929 MRI image + expert-annotated mask pairs

Dataset characteristics:

| Property | Value |
|---|---|
| Scans with tumor | 1,373 (34.9%) |
| Scans without tumor | 2,556 (65.1%) |
| Average tumor area | 2.9% of image |
| Smallest tumor | 0.01% of image |
| Largest tumor | 12.1% of image |

The 65% no-tumor ratio represents a real clinical characteristic — many MRI slices show healthy tissue above or below the tumor location.

---

## ⚠️ Handling Class Imbalance

This dataset has two levels of imbalance:

1. **Scan level:** 65% of scans have no tumor at all
2. **Pixel level:** Even in tumor scans, only ~3% of pixels are tumor

**Solutions applied:**

- `BCEWithLogitsLoss(pos_weight=3.0)` — tumor pixel errors penalized 3× more than background
- `Dice Loss` — directly measures mask overlap, unaffected by background dominance
- **Combined loss = BCE + Dice** — pixel confidence + shape overlap quality

---

## 🔧 Training Configuration

| Parameter | Value |
|---|---|
| Architecture | U-Net + VGG16 encoder |
| Input size | 256 × 256 |
| Batch size | 16 |
| Epochs | 20 (interrupted from 25) |
| Encoder LR | 0.00001 (pretrained — slow) |
| Decoder LR | 0.0005 (new — fast) |
| Scheduler | CosineAnnealingLR |
| Early stopping | Patience 7 epochs |
| Loss function | BCE (pos_weight=3.0) + Dice |
| Optimizer | Adam |

**Differential Learning Rates** — key technique used:
- Encoder uses very small LR to preserve pretrained ImageNet features
- Decoder uses higher LR as it learns from scratch

---

## 📈 Final Model Performance

| Metric | Score |
|---|---|
| **Validation IoU** | **71.1%** |
| **Validation Dice** | **78.2%** |
| **Test IoU** | **73.9%** |
| **Test Dice** | **81.2%** |
| Best individual case IoU | 97.0% |

**Benchmark context:**
Published papers on LGG segmentation with similar architectures report **0.78–0.86 Dice Score**.
This model's **81.2% Dice** places it within the published research range at 20 epochs.

---

## 🚀 ML Pipeline Overview

```
LGG Brain MRI Dataset (kaggle_3m)
↓
Image-Mask Pair Loading (.tif format)
↓
Dataset Analysis (tumor balance, size distribution)
↓
Synchronized Augmentation (image + mask together)
↓
BrainSegDataset class (custom PyTorch Dataset)
↓
80 / 10 / 10 Train / Val / Test Split
↓
U-Net + VGG16 model (Transfer Learning)
↓
BCE + Dice combined loss training
↓
Differential learning rate optimization
↓
Early stopping + best model checkpointing
↓
IoU + Dice Score evaluation
↓
Visual prediction analysis (best/worst cases)
↓
Flask web application deployment
```

---

## 🌐 Web Application

A complete Flask deployment provides a clinical-grade interface:

### LINK: https://muhammedpanchla-neuroscan-ai.hf.space/#

**Features:**
- Drag and drop MRI scan upload (JPEG, PNG, TIFF)
- Real-time U-Net segmentation inference
- 3-panel output: MRI scan · Tumor mask · Red overlay
- Model confidence heatmap (pixel-wise probability)
- Tumor coverage percentage + pixel count
- Tumor Detected / No Tumor verdict with color coding

---

## 📁 Repository Structure

```
brain-tumor-segmentation/
│
├── app/
│   ├── app.py                              ← Flask application
│   └── templates/
│       └── index.html                      ← Web interface
│
├── model/
│   └── brain_tumor_segmentation_best_model.pth   ← Saved checkpoint
│
├── notebooks/
│   └── BTSegment_Final.ipynb               ← Complete training notebook
│
├── static/
│   └── uploads/                            ← Auto-created for uploads
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

**Deep Learning**
- PyTorch
- segmentation-models-pytorch (U-Net implementation)

**Medical Imaging**
- tifffile (TIFF medical image reading)
- albumentations (synchronized image + mask augmentation)

**Data Processing**
- NumPy
- Pandas

**Visualization**
- Matplotlib

**Deployment**
- Flask
- PIL (Pillow)

---

## 🔬 Technical Highlights

This project demonstrates advanced deep learning engineering:

- Complete end-to-end segmentation pipeline in PyTorch
- Transfer Learning from ImageNet → medical imaging domain
- U-Net architecture with skip connections for spatial precision
- Synchronized image + mask augmentation using albumentations
- Combined BCE + Dice loss for imbalanced medical data
- Differential learning rates for encoder/decoder
- Cosine annealing LR scheduling
- Early stopping with best model checkpointing
- IoU and Dice Score evaluation (medical AI standard metrics)
- Base64 image encoding for Flask API response
- Clinical-grade web interface with real-time inference

---

## 🏥 Business and Clinical Impact

| Stakeholder | Benefit |
|---|---|
| Radiologists | Reduce MRI analysis time from 45 min → under 5 min |
| Surgeons | Precise tumor boundary for pre-surgical planning |
| Hospitals | Process more patients with same resources |
| Patients | Faster diagnosis, earlier treatment |

---

## 📋 Evaluation Methodology

- **IoU (Intersection over Union)** — standard segmentation metric
- **Dice Score** — primary metric in medical AI literature
- Empty masks correctly excluded from metric calculation
- Best/worst case analysis performed on test set
- Visual overlay comparison (red = missed, green = extra, yellow = correct)

---

## 🎯 Future Improvements

- ResNet34 or EfficientNet-B4 encoder — expected +2–3% IoU over VGG16
- 512×512 input resolution — better detection of small tumors
- Test Time Augmentation (TTA) — average predictions for robustness
- 3D U-Net — process full volumetric MRI stacks
- Multi-class segmentation — distinguish tumor core, enhancing region, and edema

---

## 👨‍💻 Author

**Mohammed Panchla**

Aspiring Machine Learning Engineer focused on production-ready AI systems and healthcare deep learning.

---

## ⭐ If you found this project useful, consider giving it a star!
