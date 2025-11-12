import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import time

# ==============================
# PAGE SETUP
# ==============================
st.set_page_config(page_title="Chest X-Ray Classifier", page_icon="🩻", layout="centered")

st.title("🧠 Chest X-Ray Pneumonia Classifier")
st.markdown("""
Upload a chest X-ray image and the model will predict whether it shows **Pneumonia** or **Normal** lungs.  
*(Using Baseline CNN with Fast Occlusion Visualization)*
""")

# ==============================
# LOAD MODEL
# ==============================
MODEL_PATH = "models/baseline_cnn.h5"

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

try:
    model = load_cnn_model()
    st.success("✅ Baseline CNN model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ==============================
# IMAGE UPLOAD
# ==============================
uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🩻 Uploaded X-ray", use_container_width=True)

    # preprocess
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.write("🔍 Running diagnosis...")
    prediction = model.predict(img_array, verbose=0)[0][0]

    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction) * 100

    if label == "PNEUMONIA":
        st.error(f"🦠 **Prediction:** {label}  \n**Confidence:** {confidence:.2f}%")
    else:
        st.success(f"💨 **Prediction:** {label}  \n**Confidence:** {confidence:.2f}%")

    # ==============================
    # FAST OCCLUSION MAP
    # ==============================
    with st.expander("Show Occlusion Sensitivity Map (Explainability)"):
        st.write("🧠 Computing fast occlusion map (~5 seconds)...")

        def fast_occlusion_sensitivity(model, image, patch_size=25, stride=25):
            """Fast occlusion map with coarse grid and smooth interpolation."""
            h, w, _ = image.shape
            grid_h = (h - patch_size) // stride + 1
            grid_w = (w - patch_size) // stride + 1

            heatmap = np.zeros((grid_h, grid_w))
            base_pred = model.predict(image[np.newaxis, ...])[0][0]

            total = grid_h * grid_w
            progress = st.progress(0)
            k = 0

            for i in range(grid_h):
                for j in range(grid_w):
                    img_copy = np.array(image, copy=True)
                    y, x = i * stride, j * stride
                    img_copy[y:y+patch_size, x:x+patch_size, :] = 0
                    new_pred = model.predict(img_copy[np.newaxis, ...])[0][0]
                    heatmap[i, j] = abs(base_pred - new_pred)

                    k += 1
                    progress.progress(k / total)

            heatmap = cv2.resize(heatmap, (224, 224))
            heatmap = cv2.GaussianBlur(heatmap, (17, 17), 0)
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) + 1e-8)
            return heatmap

        img_norm = np.array(img_resized) / 255.0
        start = time.time()
        heatmap = fast_occlusion_sensitivity(model, img_norm)
        end = time.time()

        st.caption(f"✅ Done in {end - start:.1f} seconds")

        overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(np.array(img_resized), 0.7, overlay, 0.3, 0)

        st.image(blended, caption="✨ Smooth Occlusion Sensitivity Map", use_container_width=True)

else:
    st.info("👆 Upload an image to start the prediction.")

# ==============================
# FOOTER
# ==============================
st.markdown("""
---
**Developed by:** Vanuj Gangrade  
🧠 *Medical Image Classification using Baseline CNN + Occlusion Visualization*
""")
