import sys
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QInputDialog
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from face_alignment import annotate_face_alignment
from user_recognition import UserRecognition


class CameraWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Project")
        self.resize(900, 700)

        self.image_label = QLabel("Camera not started")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.user_label = QLabel("User: Unknown User")
        self.user_label.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.save_user_button = QPushButton("Save Current User")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.user_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.save_user_button)
        self.setLayout(layout)

        self.cap = None
        self.current_frame = None
        self.user_recognition = UserRecognition()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.save_user_button.clicked.connect(self.save_current_user)

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        if self.cap.isOpened():
            self.timer.start(30)
            self.image_label.setText("Camera started")
            self.user_label.setText("User: Unknown User")
        else:
            self.image_label.setText("Could not access camera")

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.image_label.setText("Camera stopped")
        self.user_label.setText("User: Unknown User")

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.image_label.setText("Failed to read frame")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = frame.copy()
        user_name = self.user_recognition.recognize(frame)
        self.user_label.setText(f"User: {user_name}")

        frame = annotate_face_alignment(frame)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap)

    def save_current_user(self):
        if self.current_frame is None:
            self.user_label.setText("Start the camera before saving a user.")
            return

        name, accepted = QInputDialog.getText(self, "Save User", "User name:")
        if not accepted:
            return

        try:
            self.user_recognition.save_user(name, self.current_frame)
        except ValueError as error:
            self.user_label.setText(str(error))
            return

        self.user_label.setText(f"Saved user: {name.strip()}")

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

    def RecognizeFace(self, event):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraWindow()
    window.show()
    sys.exit(app.exec())
