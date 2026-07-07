from pathlib import Path

import cv2
import numpy as np


USER_IMAGE_DIR = Path(__file__).resolve().parent / "user_images"
UNKNOWN_USER = "Unknown User"
IMAGE_SIZE = (120, 120)
MATCH_THRESHOLD = 0.55

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class UserRecognition:
    def __init__(self, user_image_dir=USER_IMAGE_DIR):
        self.user_image_dir = Path(user_image_dir)
        self.user_image_dir.mkdir(exist_ok=True)
        self.known_users = []
        self.reload_users()

    def reload_users(self):
        self.known_users = []

        for image_path in sorted(self.user_image_dir.glob("*.jpg")):
            face_image = cv2.imread(str(image_path))
            if face_image is None:
                continue

            face_template = self._prepare_face(face_image)
            if face_template is None:
                continue

            self.known_users.append(
                {
                    "name": self._name_from_filename(image_path.stem),
                    "template": face_template,
                }
            )

    def recognize(self, rgb_frame):
        if not self.known_users:
            return UNKNOWN_USER

        face = self._extract_face(rgb_frame)
        if face is None:
            return UNKNOWN_USER

        best_name = UNKNOWN_USER
        best_score = float("inf")

        for user in self.known_users:
            score = self._compare_faces(face, user["template"])
            if score < best_score:
                best_score = score
                best_name = user["name"]

        if best_score > MATCH_THRESHOLD:
            return UNKNOWN_USER

        return best_name

    def save_user(self, name, rgb_frame):
        clean_name = self._clean_name(name)
        if not clean_name:
            raise ValueError("Please enter a user name.")

        face = self._extract_face(rgb_frame)
        if face is None:
            raise ValueError("No face found. Align your face in the camera and try again.")

        image_path = self.user_image_dir / f"{clean_name}.jpg"
        cv2.imwrite(str(image_path), cv2.cvtColor(face, cv2.COLOR_GRAY2BGR))
        self.reload_users()
        return image_path

    def _extract_face(self, rgb_frame):
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        faces = _FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        if len(faces) == 0:
            return None

        x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
        face = gray[y : y + height, x : x + width]
        return self._prepare_face(face)

    def _prepare_face(self, face_image):
        if len(face_image.shape) == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        face_image = cv2.resize(face_image, IMAGE_SIZE)
        face_image = cv2.equalizeHist(face_image)
        return face_image

    def _compare_faces(self, face_a, face_b):
        difference = cv2.absdiff(face_a, face_b)
        return float(np.mean(difference) / 255.0)

    def _clean_name(self, name):
        return "".join(
            char for char in name.strip().replace(" ", "_") if char.isalnum() or char == "_"
        )

    def _name_from_filename(self, filename):
        return filename.replace("_", " ")
