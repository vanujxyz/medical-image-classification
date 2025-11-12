<div align="center">

<h1 style="font-size: 32px; color: #0f172a; margin-bottom: 10px;">
🩺 Medical Chest X-Ray Classification
</h1>

<p style="font-size: 16px; color: #334155; max-width: 700px; margin: auto;">
A compact, reproducible deep learning project for detecting <b>Pneumonia</b> in chest X-rays using a <b>Baseline CNN</b> model.  
Includes training notebooks, evaluation results, and an interactive <b>Streamlit web app</b> with explainable AI visualization.
</p>

<br>

<img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square">
<img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square">
<img src="https://img.shields.io/badge/Streamlit-App-red?style=flat-square">
<img src="https://img.shields.io/badge/Jupyter-Notebook-yellow?style=flat-square">

</div>

---

## 🚀 Quick Start

Run these commands from the project root:

```bash
# 1️⃣ Activate virtual environment (if not active)
.venv\Scripts\activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run Streamlit app
streamlit run app/app.py

# 4️⃣ Open the evaluation notebook
jupyter lab notebooks/05_model_evaluation.ipynb
```

> 💡 **Tip:** The app may take a few seconds to start the first time — TensorFlow initializes the model in memory.

---

## 📁 Project Structure

```
medical-image-classification/
│
├── app/                     # Streamlit demo (app.py)
├── models/                  # Saved model files (e.g. baseline_cnn.h5)
├── data/                    # Dataset (train / val / test)
├── notebooks/               # Training & evaluation notebooks
├── results/                 # Plots, confusion matrices, etc.
├── src/                     # Optional helper scripts
├── requirements.txt         # Dependencies
└── README.md                # You are here
```

---

## 🧠 Models and Dataset

### 🔹 Baseline CNN
A lightweight CNN trained from scratch — fast, compact, and accurate (≈ 86% test accuracy).  
Used by default in the Streamlit app.

### 🔹 EfficientNet-B0 (Transfer Learning)
Fine-tuned version of EfficientNet for higher accuracy.  
Training & fine-tuning are included in the notebooks, not the live app (to save time).

### 📂 Dataset
Expected structure:
```
data/chest_xray/
 ├── train/
 ├── val/
 └── test/
```

> If your dataset path differs, update it in the notebooks or adjust the code at the top using `os.chdir()`.

---

## ⚙️ How It Works

1. Upload a chest X-ray image (`.jpg` / `.png`)
2. The image is resized to **224×224**, normalized, and passed into the Baseline CNN.
3. The model predicts a probability → **Normal** or **Pneumonia**.
4. The app visualizes **Occlusion Sensitivity** — showing regions that most affect the model’s decision.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Chest_Xray_PA_3-8-2010.png/400px-Chest_Xray_PA_3-8-2010.png" width="300" />
</p>

> 🧩 The occlusion map provides explainability by highlighting regions that influence predictions.

---

## 📊 Notebooks Overview

| Notebook | Description |
|-----------|--------------|
| **01_data_exploration.ipynb** | Explore dataset structure & sample X-rays |
| **02_train_baseline.ipynb** | Train the Baseline CNN & save `baseline_cnn.h5` |
| **03_train_transfer.ipynb** | EfficientNet-B0 fine-tuning |
| **05_model_evaluation.ipynb** | Compare Baseline vs. EfficientNet (classification report, confusion matrix, ROC) |

---

## 🧾 Results (Sample)

| Model | Accuracy | Loss | Notes |
|--------|-----------|------|-------|
| **Baseline CNN** | 86.1% | 0.32 | Fast & light, used in app |
| **EfficientNet-B0 (transfer)** | ~94% | 0.19 | Trained longer for best performance |

---

## 🧩 Streamlit App Features

✅ Upload any X-ray image  
✅ Get prediction (Normal / Pneumonia)  
✅ Visualize Occlusion Sensitivity heatmap  
✅ Smooth, fast visualization (≈ 5 seconds)  
✅ Works offline with pre-trained CNN  

---

## 🛠 Troubleshooting

| Issue | Cause / Fix |
|-------|--------------|
| App shows blank page | Wait 30–60s — TensorFlow is loading |
| Missing model file | Ensure `models/baseline_cnn.h5` exists |
| Wrong results | Confirm same preprocessing as during training |
| Push errors on GitHub | Large model files — add to `.gitignore` or use Git LFS |

---

## ⏱ Useful Commands

```bash
# List model files
python -c "import os; print(os.listdir('models'))"

# Run demo
streamlit run app/app.py

# Run evaluation notebook
jupyter lab notebooks/05_model_evaluation.ipynb
```

---

## 👨‍💻 Author

**Vanuj Gangrade**  
Deep Learning | Computer Vision | AI Explainability  
📧 *[add your contact email or LinkedIn/GitHub link]*

---

<p align="center" style="color: #6b7280; font-size: 13px;">
💡 *This project was developed as part of a deep learning coursework — focused on improving medical image classification and explainability.*
</p>
