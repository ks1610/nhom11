import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn as nn
from torchvision import models
import numpy as np
import joblib

# ======================================================
# CONFIG
# ======================================================
RF_MODEL_PATH = "randomforest_best.pkl"
SCALER_PATH = "scaler.pkl"

# ======================================================
# LOAD MODELS (CACHED)
# ======================================================
@st.cache_resource
def get_rf_model():
    rf = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Random Forest model loaded.")
    return rf, scaler

@st.cache_resource
def get_feature_extractor():
    """
    Uses pretrained ResNet18 as a feature extractor (removes last FC layer)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return feature_extractor, transform, device

# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_feature(image_pil, feature_extractor, transform, device):
    img = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_extractor(img)
    return feat.cpu().numpy().flatten()

# ======================================================
# RANDOM FOREST PREDICT
# ======================================================
def predict_random_forest(rf, scaler, feature_extractor, transform, device, image_pil):
    vec = extract_feature(image_pil, feature_extractor, transform, device)
    vec_scaled = scaler.transform([vec])
    pred = rf.predict(vec_scaled)[0]
    prob = rf.predict_proba(vec_scaled)[0]

    label_map = {0: "Non-Defective", 1: "Defective"}
    return label_map[pred], prob[pred] * 100

# ======================================================
# STREAMLIT UI
# ======================================================
st.set_page_config(layout="wide", page_title="Phone Defect Detection (Random Forest)")

st.title("📱 Phone Defect Detection Web App")
st.write("Ứng dụng dựa trên **Random Forest + ResNet18** để phát hiện lỗi màn hình điện thoại.")

# Load all models
rf_model, scaler = get_rf_model()
feature_extractor, transform, device = get_feature_extractor()

uploaded_file = st.file_uploader("📤 Tải lên ảnh điện thoại", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh gốc", use_container_width=True)

    with st.spinner("🔍 Đang dự đoán..."):
        rf_label, rf_conf = predict_random_forest(
            rf_model, scaler, feature_extractor, transform, device, image
        )

    st.markdown("### 🔎 Kết quả dự đoán:")
    color = "red" if rf_label == "Defective" else "green"
    st.markdown(f"<h3 style='color:{color}'>{rf_label}</h3>", unsafe_allow_html=True)
    st.write(f"Độ tin cậy: **{rf_conf:.2f}%**")

else:
    st.info("Hãy tải lên một ảnh để bắt đầu dự đoán.")
