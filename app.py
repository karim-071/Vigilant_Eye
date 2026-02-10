import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from src.models.efficientnet import get_model

st.set_page_config(page_title="VigilantEye", layout="centered")
st.title("VigilantEye – Deepfake Detector")

device = torch.device("cpu")

model = get_model()
model.load_state_dict(torch.load("weights/vigilant_eye.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    label = "REAL" if prob > 0.5 else "FAKE"

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: **{prob:.2f}**")

