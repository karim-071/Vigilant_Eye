import torch
import cv2
from PIL import Image
from src.inference.preprocess import transform

# Face detector (OpenCV)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def crop_face(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        image_bgr = image_bgr[y:y+h, x:x+w]

    return image_bgr


def predict_image(model, image_path, device="cpu", threshold=0.4):

    model.eval()

    # Read using OpenCV
    image = cv2.imread(image_path)

    # Face crop
    image = crop_face(image)

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    label = 1 if prob > threshold else 0

    return prob, label
