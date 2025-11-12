import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import matplotlib.pyplot as plt

# ==============================================================
# Load model
# ==============================================================
@st.cache_resource
def load_trained_model():
    model_path = r"C:\Users\vgang\medical-image-classification\models\efficientnet_b0_finetuned.keras"
    model = load_model(model_path)
    return model

model = load_trained_model()

# ==============================================================
# Occlusion sensitivity function
# ==============================================================
def occlusion_map(image_array, model, patch_size=32, stride=16):
    img_h, img_w = image_array.shape[:2]
    arr = image_array / 255.0
    inp = np.expand_dims(arr, axis=0).astype(np.float32)
    base_pred = float(model.predict(inp, verbose=0)[0][0])
    h_steps = (img_h - patch_size) // stride + 1
    w_steps = (img_w - patch_size) // stride + 1
    heatmap = np.zeros((h_steps, w_steps), dtype=np.float32)

    for i in range(h_steps):
        row_imgs = []
        positions = []
        for j in range(w_steps):
            y, x = i * stride, j * stride
            occluded = arr.copy()
            occluded[y:y+patch_size, x:x+patch_size, :] = np.mean(arr, axis=(0,1))
            row_imgs.append(occluded)
            positions.append((i,j))
        batch = np.stack(row_imgs, axis=0)
        preds = model.predict(batch, verbose=0).squeeze()
        effects = base_pred - preds
        for (i_pos, j_pos), eff in zip(positions, effects):
            heatmap[i_pos, j_pos] = eff

    heatmap = cv2.resize(heatmap, (img_w, img_h))
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor((arr*255).astype(np.uint8), cv2.COLOR_RGB2BGR), 
                              0.7, heatmap_color, 0.3, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return base_pred, overlay

# ==============================================================
# Streamlit UI
# ==============================================================
st.set_page_config(page_title="Pneumonia Detection App", page_icon="🩻", layout="wide")

st.title("🩺 Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image and see the model's prediction with an explainability heatmap.")

uploaded_file = st.file_uploader("Upload a chest X-ray image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    # Load image
    image = Image.open(temp_file.name).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image)

    st.image(image, caption="Uploaded X-ray", use_column_width=False, width=300)

    with st.spinner("Analyzing X-ray... please wait ⏳"):
        pred_score, overlay = occlusion_map(image_array, model)
        label = "PNEUMONIA" if pred_score > 0.5 else "NORMAL"
        confidence = pred_score if pred_score > 0.5 else 1 - pred_score

    st.success(f"**Prediction: {label}** (confidence: {confidence:.2%})")

    st.subheader("Explainability Heatmap")
    st.image(overlay, caption="Occlusion Sensitivity Map", use_column_width=True)

st.markdown("---")
st.markdown("Developed by **Vanuj Gangrade ,Agathian ganesan,Ardra Haridas,Joshua Varkey** | AI Medical Imaging Project 🧠")
