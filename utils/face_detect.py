import cv2
from mtcnn import MTCNN

detector = MTCNN()

def detect_face(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]['box']

    # Ensure coordinates are positive
    x, y = max(0, x), max(0, y)

    face = image[y:y+h, x:x+w]
    return face
