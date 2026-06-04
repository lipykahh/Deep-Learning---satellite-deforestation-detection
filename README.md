# Deep-Learning---satellite-deforestation-detection
Binary satellite image classifier to detect deforestation using ResNet18, EuroSAT, and Grad-CAM — aligned with UN SDG 15 (Life on Land)
# 🌿 Satellite-Based Deforestation Detection

> Binary image classifier for automated forest monitoring from Sentinel-2 satellite imagery.  
> Aligned with **UN SDG 15 – Life on Land** (Primary) and **SDG 13 – Climate Action** (Secondary)

---

## 📌 Overview

Deforestation is one of the leading drivers of biodiversity loss and climate change.
This project builds an automated deep learning pipeline to classify satellite image patches as
**Forest** or **Non-Forest** using the EuroSAT dataset and a pretrained ResNet18 model —
making large-scale forest monitoring faster, cheaper, and accessible without field surveys.

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| Accuracy | 99.89% |
| Precision | 99.78% |
| Recall | 100.00% |
| F1-Score | 99.89% |
| Test Images | 900 |
| Misclassified | 1 |

---

## 🗂️ Dataset

- **Source:** [EuroSAT](https://github.com/phelber/EuroSAT) — Sentinel-2 satellite imagery (ESA)
- **Original:** 27,000 images across 10 land-use classes
- **Converted to binary:** Forest vs Non-Forest
- **Class balancing:** Random undersampling (3,000 each → 6,000 total)
- **Split:** 70% train / 15% val / 15% test

---

## 🧠 Model

- **Architecture:** ResNet18 pretrained on ImageNet
- **Modification:** Final FC layer → `Linear(512, 1)` + Sigmoid
- **Loss:** BCEWithLogitsLoss
- **Optimizer:** Adam (lr = 1e-4)
- **Epochs:** 15 | **Batch size:** 32

---

## 🔍 Explainability

Grad-CAM applied to `model.layer4[-1]` to visualize model attention:
- ✅ Correctly classified Forest images — activation over tree canopy
- ❌ Misclassified example — diffused activation over ambiguous vegetation
- 🏗️ Non-Forest samples — activation over built-up/bare regions

---

## 🚀 Deployment

Simple Streamlit web app — upload any satellite image patch and get:
- Forest / Non-Forest prediction
- Confidence score
- Grad-CAM heatmap overlay

---

## 📁 Project Structure

```
satellite-deforestation-detection/
├── app.py                  # Streamlit deployment
├── src/
│   ├── prepare_data.py     # Dataset prep & balancing
│   ├── train.py            # Model training
│   ├── evaluate.py         # Metrics & confusion matrix
│   └── gradcam.py          # Grad-CAM explainability
├── outputs/
│   ├── metrics/            # Loss curve, confusion matrix
│   └── gradcam/            # Heatmap images
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Evaluate on test set
python src/evaluate.py

# Generate Grad-CAM heatmaps
python src/gradcam.py

# Launch Streamlit app
streamlit run app.py
```

> **Note:** This project was developed on Google Colab with T4 GPU and Google Drive storage.
> Adjust file paths in each script if running locally.

---

## 🌍 SDG Alignment

| SDG | Role |
|-----|------|
| **SDG 15 – Life on Land** (Primary) | Automates forest cover monitoring at scale |
| **SDG 13 – Climate Action** (Secondary) | Enables early detection of forest loss for climate intervention |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-brightgreen)
![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-orange)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
