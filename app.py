import streamlit as st
import torch
import tempfile
from PIL import Image
import os

from models.model import get_model
from src.inference.predict_image import predict_image
from src.inference.predict_video import predict_video


# PAGE CONFIG
st.set_page_config(
    page_title="Vigilant Eye",
    page_icon="👁️",
    layout="centered"
)

st.title("Vigilant Eye — Deepfake Detector")

# DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD MODEL (CACHE)
@st.cache_resource
def load_model():
    model = get_model(
        weight_path="weights/vigilant_eye.pth",
        device=device
    )
    return model

model = load_model()


# MODE SELECTION
mode = st.radio(
    "Choose detection type:",
    ["Image Detection", "Video Detection"]
)

THRESHOLD = 0.7


# IMAGE DETECTION
if mode == "Image Detection":

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:

        # Show image
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", width="stretch")

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        with st.spinner("Analyzing image..."):
            prob, label = predict_image(
                model,
                temp_path,
                device=device,
                threshold=THRESHOLD
            )

        os.remove(temp_path)

        st.divider()

        if prob > THRESHOLD:
            confidence = prob
            st.error(f"FAKE ({confidence*100:.2f}% confidence)")
        else:
            confidence = 1 - prob
            st.success(f"REAL ({confidence*100:.2f}% confidence)")

        st.progress(float(confidence))


# VIDEO DETECTION
else:

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        # Save safely using temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        st.video(video_path)

        with st.spinner("Processing video frames..."):
            prob, label = predict_video(
                model,
                video_path,
                device=device,
                frame_step=5,
                threshold=THRESHOLD
            )

        os.remove(video_path)

        if prob is None:
            st.warning("No detectable face frames found.")
        else:

            st.divider()

            confidence = prob if label == 1 else (1 - prob)

            if label == 1:
                st.error(f"FAKE VIDEO ({confidence*100:.2f}% confidence)")
            else:
                st.success(f"REAL VIDEO ({confidence*100:.2f}% confidence)")


            st.progress(float(confidence))
