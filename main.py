import sys

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QInputDialog, QVBoxLayout, QWidget

from barcode_scanner import BarcodeScanner
from face_alignment import annotate_face_alignment
from medicine_registry import MedicineRegistry
from user_recognition import UNKNOWN_USER, UserRecognition


class CameraWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Project")
        self.resize(900, 700)

        self.image_label = QLabel("Camera not started")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.user_label = QLabel("User: Unknown User")
        self.user_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("Status: Align your face inside the box.")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.medicine_label = QLabel("Medicine information will appear here after a barcode is scanned.")
        self.medicine_label.setAlignment(Qt.AlignCenter)
        self.medicine_label.setWordWrap(True)

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.save_user_button = QPushButton("Save Current User")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.user_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.medicine_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.save_user_button)
        self.setLayout(layout)

        self.cap = None
        self.current_frame = None
        self.user_recognition = UserRecognition()
        self.barcode_scanner = BarcodeScanner()
        self.medicine_registry = MedicineRegistry()
        self.active_user_name = UNKNOWN_USER
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
            self.user_label.setStyleSheet("")
            self.status_label.setText("Status: Align your face inside the box.")
            self.status_label.setStyleSheet("")
            self.medicine_label.setText(
                "Medicine information will appear here after a barcode is scanned."
            )
            self.active_user_name = UNKNOWN_USER
        else:
            self.image_label.setText("Could not access camera")

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.image_label.setText("Camera stopped")
        self.user_label.setText("User: Unknown User")
        self.user_label.setStyleSheet("")
        self.status_label.setText("Status: Camera stopped.")
        self.status_label.setStyleSheet("")
        self.medicine_label.setText("Medicine information will appear here after a barcode is scanned.")
        self.active_user_name = UNKNOWN_USER

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.image_label.setText("Failed to read frame")
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = rgb_frame.copy()
        if self.active_user_name == UNKNOWN_USER:
            detected_user_name = self.user_recognition.recognize(rgb_frame)
            if detected_user_name != UNKNOWN_USER:
                self.active_user_name = detected_user_name

        user_name = self.active_user_name
        user_recognized = user_name != UNKNOWN_USER
        self.user_label.setText(f"User: {user_name}")
        self.user_label.setStyleSheet("color: green;" if user_recognized else "")

        barcode_results = self.barcode_scanner.scan(frame) if user_recognized else []
        barcode_points = [result["points"] for result in barcode_results if result["points"]]
        self._update_status_and_medicine(user_name, barcode_results)

        annotated_frame = annotate_face_alignment(
            rgb_frame,
            user_recognized=user_recognized,
            barcode_points_list=barcode_points,
        )
        h, w, ch = annotated_frame.shape
        bytes_per_line = ch * w
        image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
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

    def _update_status_and_medicine(self, user_name, barcode_results):
        if user_name == UNKNOWN_USER:
            self.status_label.setText("Status: Align your face inside the box for recognition.")
            self.status_label.setStyleSheet("")
            self.medicine_label.setText(
                "Medicine information will appear here after a barcode is scanned."
            )
            return

        self.status_label.setText(f"Status: {user_name} recognized. Scan the barcode.")
        self.status_label.setStyleSheet("color: green;")

        self.medicine_registry.reload()

        for barcode_result in barcode_results:
            record = self.medicine_registry.find_for_user(barcode_result["value"], user_name)
            if record is not None:
                self.medicine_label.setText(self.medicine_registry.format_record_summary(record))
                return

            owner_name = self.medicine_registry.find_owner_for_barcode(barcode_result["value"])
            if owner_name is not None:
                self.medicine_label.setText(
                    f"Scanned barcode belongs to {owner_name}, not {user_name}."
                )
                return

        if barcode_results:
            self.medicine_label.setText("Barcode scanned, but no medicine record was found for this user.")
            return

        self.medicine_label.setText("Medicine information will appear here after a barcode is scanned.")

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
