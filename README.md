# 🌿 Satellite-Based Deforestation Detection

> Automated binary classification of satellite image patches into **Forest** or **Non-Forest**
> using a fine-tuned ResNet18 and Grad-CAM explainability.
> Aligned with **UN SDG 15 – Life on Land** (Primary) and **SDG 13 – Climate Action** (Secondary).

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | **99.89%** |
| Precision | **99.78%** |
| Recall | **100.00%** |
| F1-Score | **99.89%** |
| Test images | 900 |
| Misclassified | 1 |

---

## 🗂️ Dataset

- **Source:** [EuroSAT](https://github.com/phelber/EuroSAT) — Sentinel-2 satellite imagery by ESA
- **Original:** 27,000 images · 10 land-use classes
- **Converted:** Binary — Forest (label 1) vs Non-Forest (label 0)
- **Balanced:** Random undersampling → 3,000 Forest + 3,000 Non-Forest = **6,000 total**
- **Split:** 70% train (4,200) · 15% val (900) · 15% test (900)

---

## 🧠 Model Architecture

| Property | Detail |
|----------|--------|
| Base model | ResNet18 (pretrained on ImageNet) |
| Modification | Final FC → `Linear(512, 1)` |
| Activation | Sigmoid |
| Loss | `BCEWithLogitsLoss` |
| Optimizer | Adam · lr = 1e-4 |
| Epochs | 15 · Batch size 32 |

---

## 📁 Project Structure

```
satellite-deforestation-detection/
├── DL_Proj_SGD.ipynb       ← Full pipeline (all 9 steps in one notebook)
├── app.py                  ← Streamlit deployment app
├── requirements.txt
└── README.md
```

> **Note:** Model weights (`best_model.pth`) and output images are not included.
> Run the notebook to generate them — they save directly to your Google Drive.

---

## 🚀 How to Run

### 1. Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/satellite-deforestation-detection/blob/main/DL_Proj_SGD.ipynb)

> Go to **Runtime → Change runtime type → T4 GPU** before running.

### 2. Set up your dataset
Place EuroSAT in Google Drive at:
```
MyDrive/
└── Eurosat/
    ├── Forest/
    ├── AnnualCrop/
    ├── Highway/
    └── ... (10 classes)
```

### 3. Add your ngrok token
In Colab: **🔑 Secrets → Add → Name: `NGROK_TOKEN` → Value: your token**

Get a free token at [dashboard.ngrok.com](https://dashboard.ngrok.com)

### 4. Run all cells in order

| Cell | Step |
|------|------|
| Cell 0 | Mount Google Drive |
| Step 1 | Folder setup |
| Step 2 | Dataset loading + class balancing |
| Step 3 | DataLoaders + transforms |
| Step 4 | ResNet18 model setup |
| Step 5 | Training loop (saves best model to Drive) |
| Step 6 | Evaluation — metrics + confusion matrix |
| Step 7 | Grad-CAM heatmaps |
| Step 8 | Streamlit app launch via ngrok |

---

## 🔍 Explainability — Grad-CAM

Grad-CAM is applied to `model.layer4[-1]` (last convolutional block of ResNet18)
to visualize which regions of the satellite image drove the prediction.

- ✅ **Correct Forest** — activation concentrated over dense tree canopy
- 🏗️ **Non-Forest** — activation over built-up or bare land regions  
- ❌ **Misclassified** — diffused attention over ambiguous vegetation boundary

---

## 🌐 Streamlit App

Upload any satellite image patch and instantly get:
- 🌲 Forest / 🏗️ Non-Forest prediction
- Confidence score + progress bar
- Grad-CAM heatmap overlay side-by-side

Runs on Colab via ngrok tunnel — no local server needed.

---

## ⚙️ Requirements

```txt
torch>=2.0
torchvision>=0.15
scikit-learn
matplotlib
seaborn
numpy
Pillow
grad-cam
streamlit
pyngrok
tqdm
```

Install:
```bash
pip install -r requirements.txt
```

---

## 🌍 SDG Alignment

| SDG | Role |
|-----|------|
| **SDG 15 – Life on Land** · Primary | Automates forest cover monitoring from satellite imagery at scale |
| **SDG 13 – Climate Action** · Secondary | Forests are carbon sinks — early loss detection enables faster climate intervention |

---

## ⚠️ Important — Before Cloning

- **Never hardcode your ngrok token** in the notebook. Use Colab Secrets instead:
  ```python
  import os
  NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")
  ```
- Model weights are excluded from this repo (too large). They save to your Drive after training.
- All Drive paths use `MyDrive/deforestation_detection/` — adjust if your structure differs.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

*Deep Learning Mini Project · B.E. AI & ML · BNMIT · VTU · 2025–2026*
*Subject: Deep Learning · Code: 23AML161*
