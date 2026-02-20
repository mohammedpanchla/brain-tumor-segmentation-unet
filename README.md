# 🧠 Brain Tumor Segmentation using Deep Learning (U-Net + VGG16 Transfer Learning)

### Clinical-Grade MRI Tumor Localization System with Flask Deployment

---

# 📌 Project Overview

This project implements a deep learning–based system for **pixel-level brain tumor segmentation** from MRI scans using a **U-Net architecture with a VGG16 encoder**.

Unlike classification models that only predict tumor presence, this system performs **precise tumor localization**, generating a binary mask that identifies the exact tumor region in each MRI image.

This project demonstrates a complete **end-to-end medical imaging AI pipeline**, including:

* MRI dataset preprocessing
* Image-mask pair handling
* Transfer learning using VGG16 encoder
* U-Net segmentation model training
* Dice Score and IoU evaluation
* Model checkpoint saving and loading
* Prediction visualization with overlays
* Flask web application deployment

---

# 🎯 Project Objective

Input:

Brain MRI Scan (256 × 256)

Output:

Binary Segmentation Mask (256 × 256)

Pixel interpretation:

* 1 → Tumor region
* 0 → Healthy tissue / background

This enables accurate tumor localization for medical and clinical analysis.

---

# 🧠 Model Architecture

Architecture: U-Net with VGG16 Encoder

U-Net consists of two main parts:

Encoder (Feature Extraction):

* VGG16 pretrained on ImageNet
* Extracts spatial and structural features
* Transfer learning improves performance and convergence speed

Decoder (Segmentation Reconstruction):

* Upsampling layers reconstruct spatial resolution
* Skip connections preserve fine tumor boundary details

Output:

* Pixel-wise tumor probability mask
* Same resolution as input image

Technical Configuration:

* Framework: PyTorch
* Input size: 256 × 256
* Output: Binary mask
* Loss function: BCEWithLogitsLoss + Dice Loss
* Optimizer: Adam
* Transfer learning: VGG16 pretrained encoder

---

# 📊 Dataset

Dataset used:

LGG Brain MRI Segmentation Dataset
Source: Kaggle (TCGA LGG)

https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

Dataset contains:

* MRI images (.tif)
* Corresponding tumor masks
* Expert annotated ground truth

Dataset characteristics:

* Total image-mask pairs: 3,929
* Patients: 110
* Tumor and non-tumor slices included

---

# 📈 Model Performance

Final evaluation metrics:

Validation IoU Score: 71.1%
Validation Dice Score: 78.2%

Test IoU Score: 73.9%
Test Dice Score: 81.2%

This performance is within the published research range for LGG tumor segmentation.

---

# 🚀 ML Pipeline

Dataset Loading
↓
Image-Mask Pair Preprocessing
↓
Train / Validation / Test Split
↓
Transfer Learning using VGG16 Encoder
↓
U-Net Model Training
↓
Loss Optimization using BCE + Dice
↓
Model Checkpoint Saving
↓
Segmentation Prediction
↓
Evaluation using IoU and Dice Score
↓
Flask Web Deployment

---

# 🌐 Web Application

The project includes a Flask web application that enables real-time tumor segmentation.

Features:

* Upload MRI scan
* Generate tumor segmentation mask
* Overlay mask visualization
* Tumor localization output

Live Demo:

https://muhammedpanchla-neuroscan-ai.hf.space/#

---

# 📁 Repository Structure

```
brain-tumor-segmentation/
│
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── model/
│   └── brain_tumor_segmentation_best_model.pth
│
├── notebooks/
│   └── BTSegment_Final.ipynb
│
├── results/
│   ├── training_loss.png
│   ├── dice_score.png
│   ├── prediction_examples.png
│   └── overlay_examples.png
│
├── test_samples/
│   ├── sample1.png
│   └── sample2.png
│
├── requirements.txt
├── README.md
```

---

# 📊 Results Visualization

The results folder contains:

* Training loss curves
* Dice score progression
* Sample segmentation predictions
* Tumor mask overlay visualizations

These confirm accurate tumor localization capability.

---

# ⚙️ Technologies Used

Deep Learning:

* PyTorch
* segmentation-models-pytorch

Medical Imaging:

* OpenCV
* tifffile
* albumentations

Data Processing:

* NumPy
* Pandas

Visualization:

* Matplotlib

Deployment:

* Flask
* Pillow

---

# 🔬 Technical Highlights

Key deep learning engineering features:

* U-Net segmentation architecture
* Transfer learning using VGG16 encoder
* Combined Dice + BCE loss optimization
* IoU and Dice Score evaluation metrics
* Medical image segmentation pipeline
* Model checkpoint saving and loading
* Real-time Flask deployment

---

# 🏥 Clinical and AI Impact

This system enables:

* Accurate tumor localization
* Faster MRI analysis
* AI-assisted radiology workflows
* Medical imaging automation

---

# 👨‍💻 Author

Mohammed Panchla

Machine Learning Engineer specializing in Medical Imaging AI

---

# ⭐ If you found this project useful, please consider giving it a star.
