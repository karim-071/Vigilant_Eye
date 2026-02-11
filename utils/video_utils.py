import cv2
import numpy as np
import torch
from torchvision import transforms
from utils.face_detect import detect_face
from utils.preprocess import preprocess_image

def extract_frames(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        if len(frames) >= num_frames:
            break

    cap.release()
    return frames


def predict_video(video_path, model):
    frames = extract_frames(video_path)

    probs = []

    model.eval()

    for frame in frames:
        face = detect_face(frame)

        if face is not None:
            img = preprocess_image(face)

            img_tensor = torch.tensor(img, dtype=torch.float32)
            img_tensor = img_tensor.permute(0, 3, 1, 2)  # NHWC → NCHW

            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output).item()

            probs.append(prob)

    if len(probs) == 0:
        return None

    return sum(probs) / len(probs)