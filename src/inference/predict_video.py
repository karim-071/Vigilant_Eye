import cv2
import torch
import numpy as np
from PIL import Image
from src.inference.preprocess import transform


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        frame = frame[y:y+h, x:x+w]

    return frame


def predict_video(model, video_path, device="cpu",
                    frame_step=5, threshold=0.4):

    model.eval()
    cap = cv2.VideoCapture(video_path)

    probs = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_step == 0:

            # Face crop
            frame = crop_face(frame)

            # Stabilize compression noise
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                prob = torch.sigmoid(output).item()
                probs.append(prob)

        frame_id += 1

    cap.release()

    if len(probs) == 0:
        return None, None

    # MEDIAN aggregation (more stable than mean)
    video_prob = float(np.median(probs))

    label = 1 if video_prob > threshold else 0

    return video_prob, label
