# import os
# import streamlit as st
# from PIL import Image
# import torch
# import torchvision.transforms as T
# import torch.nn as nn
# from torchvision import models
# import numpy as np
# import joblib

# RF_MODEL_PATH = "rf_clean.pkl"
# SCALER_PATH = "scaler_clean.pkl"

# # LOAD MODELS (CACHED)
# @st.cache_resource
# def get_rf_model():
#     rf = joblib.load(RF_MODEL_PATH)
#     scaler = joblib.load(SCALER_PATH)
#     print("✅ Random Forest model loaded.")
#     return rf, scaler


# @st.cache_resource
# def get_feature_extractor():
#     """Feature extractor using pretrained ResNet18"""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#     feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
#     feature_extractor = feature_extractor.to(device)
#     feature_extractor.eval()

#     transform = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#     ])
#     return feature_extractor, transform, device


# def extract_feature(image_pil, feature_extractor, transform, device):
#     img = transform(image_pil).unsqueeze(0).to(device)
#     with torch.no_grad():
#         feat = feature_extractor(img)
#     return feat.cpu().numpy().flatten()

# # RANDOM FOREST PREDICT
# def predict_random_forest(rf, scaler, feature_extractor, transform, device, image_pil):
#     vec = extract_feature(image_pil, feature_extractor, transform, device)
#     vec_scaled = scaler.transform([vec])
#     pred = rf.predict(vec_scaled)[0]
#     prob = rf.predict_proba(vec_scaled)[0]

#     label_map = {0: "Non-Defective", 1: "Defective", 2: "Non-Phone"}
#     pred_int = int(pred)
#     label = label_map.get(pred_int, f"Unknown ({pred_int})")
#     confidence = prob[pred_int] * 100 if pred_int < len(prob) else np.max(prob) * 100
#     return label, confidence



# # STREAMLIT UI
# st.set_page_config(layout="wide", page_title="Phone Defect Detection (Random Forest Only)")

# st.title("📱 Phone Defect Detection Web App")
# st.write("Ứng dụng **Random Forest** để phát hiện lỗi trên ảnh điện thoại.")

# # Load models
# rf_model, scaler = get_rf_model()
# feature_extractor, transform, device = get_feature_extractor()

# # UI Layout
# col1, col2 = st.columns(2)

# with col1:
#     uploaded_file = st.file_uploader("📤 Tải lên ảnh", type=["jpg", "jpeg", "png"])

# with col2:
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Ảnh gốc", use_container_width=True)

#         with st.spinner("🔍 Đang dự đoán..."):
#             rf_label, rf_conf = predict_random_forest(
#                 rf_model, scaler, feature_extractor, transform, device, image.copy()
#             )

#         st.markdown("### 🔎 Kết quả dự đoán:")
#         color = "red" if rf_label == "Defective" else "green"
#         st.markdown(f"<h4 style='color:{color}'>Kết quả: {rf_label}</h4>", unsafe_allow_html=True)
#         #st.write(f"Độ tin cậy: **{rf_conf:.2f}%**")

#     else:
#         st.info("Hãy tải lên một ảnh để bắt đầu dự đoán.")


import os
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn as nn
from torchvision import models
import numpy as np
import joblib

RF_MODEL_PATH = "rf_clean.pkl"
SCALER_PATH = "scaler_clean.pkl"

# ====================== LOAD MODELS (CACHED) ======================
@st.cache_resource
def get_rf_model():
    rf = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Random Forest model loaded.")
    return rf, scaler


@st.cache_resource
def get_feature_extractor():
    """Feature extractor using pretrained ResNet18"""
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


# ====================== FEATURE EXTRACTION ======================
def extract_feature(image_pil, feature_extractor, transform, device):
    # Apply transform (resize, tensor, normalize)
    img_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Forward pass through feature extractor
    with torch.no_grad():
        feat = feature_extractor(img_tensor)

    # Convert normalized tensor back to PIL for visualization
    img_for_display = img_tensor.squeeze(0).cpu()
    img_for_display = img_for_display * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    img_for_display = torch.clamp(img_for_display, 0, 1)
    img_for_display = T.ToPILImage()(img_for_display)

    return feat.cpu().numpy().flatten(), img_for_display


# ====================== RANDOM FOREST PREDICT ======================
def predict_random_forest(rf, scaler, feature_extractor, transform, device, image_pil):
    vec, processed_img = extract_feature(image_pil, feature_extractor, transform, device)
    vec_scaled = scaler.transform([vec])
    pred = rf.predict(vec_scaled)[0]
    prob = rf.predict_proba(vec_scaled)[0]

    label_map = {0: "Non-Defective", 1: "Defective", 2: "Non-Phone"}
    pred_int = int(pred)
    label = label_map.get(pred_int, f"Unknown ({pred_int})")
    confidence = prob[pred_int] * 100 if pred_int < len(prob) else np.max(prob) * 100
    return label, confidence, processed_img


# ====================== STREAMLIT UI ======================
st.set_page_config(layout="wide", page_title="Phone Defect Detection (Random Forest Only)")

st.title("📱 Phone Defect Detection Web App")
st.write("Ứng dụng **Random Forest** để phát hiện lỗi trên ảnh điện thoại.")

# Load models
rf_model, scaler = get_rf_model()
feature_extractor, transform, device = get_feature_extractor()

# UI Layout
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📤 Tải lên ảnh", type=["jpg", "jpeg", "png"])

with col2:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh gốc", use_container_width=True)

        with st.spinner("🔍 Đang dự đoán..."):
            rf_label, rf_conf, processed_img = predict_random_forest(
                rf_model, scaler, feature_extractor, transform, device, image.copy()
            )

        # --- Show prediction result ---
        st.markdown("### 🔎 Kết quả dự đoán:")
        color = "red" if rf_label == "Defective" else "green"
        st.markdown(f"<h4 style='color:{color}'>Kết quả: {rf_label}</h4>", unsafe_allow_html=True)
        
        # --- Show processed image ---
        st.markdown("### 🧩 Ảnh sau khi tiền xử lý:")
        st.image(processed_img, caption="Ảnh đã được Resize + Normalize", use_container_width=True)

    else:
        st.info("Hãy tải lên một ảnh để bắt đầu dự đoán.")
