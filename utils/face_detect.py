import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def detect_face(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if not results.detections:
        return None

    bbox = results.detections[0].location_data.relative_bounding_box

    h, w, _ = image.shape

    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    x, y = max(0, x), max(0, y)

    face = image[y:y+bh, x:x+bw]

    return face
