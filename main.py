import sys
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QInputDialog
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
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

        self.scan_status_label = QLabel("Recognize a saved user to begin barcode scanning.")
        self.scan_status_label.setAlignment(Qt.AlignCenter)
        self.scan_status_label.setWordWrap(True)

        self.medicine_label = QLabel("No medicine scanned.")
        self.medicine_label.setAlignment(Qt.AlignCenter)
        self.medicine_label.setWordWrap(True)

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.save_user_button = QPushButton("Save Current User")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.user_label)
        layout.addWidget(self.scan_status_label)
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
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.active_user = UNKNOWN_USER
        self.last_scan_key = None
        self.missing_user_frames = 0

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.save_user_button.clicked.connect(self.save_current_user)

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        if self.cap.isOpened():
            self.timer.start(30)
            self.image_label.setText("Camera started")
            self._set_active_user(UNKNOWN_USER)
        else:
            self.image_label.setText("Could not access camera")

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.image_label.setText("Camera stopped")
        self._set_active_user(UNKNOWN_USER)

    def update_frame(self):
        if self.cap is None:
            return

        ret, bgr_frame = self.cap.read()
        if not ret:
            self.image_label.setText("Failed to read frame")
            return

        frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self.current_frame = frame.copy()
        user_name = self.user_recognition.recognize(frame)
        self._update_active_user(user_name)
        self._scan_medicine_if_ready(bgr_frame)

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

    def _update_active_user(self, recognized_user):
        if recognized_user == self.active_user:
            self.missing_user_frames = 0
            return

        if recognized_user == UNKNOWN_USER:
            if self.active_user == UNKNOWN_USER:
                return

            self.missing_user_frames += 1
            if self.missing_user_frames >= 10:
                self._set_active_user(UNKNOWN_USER)
            return

        self.missing_user_frames = 0
        self._set_active_user(recognized_user)

    def _set_active_user(self, user_name):
        self.active_user = user_name
        self.last_scan_key = None
        self.user_label.setText(f"User: {user_name}")

        if user_name == UNKNOWN_USER:
            self.scan_status_label.setText("Recognize a saved user to begin barcode scanning.")
            self.medicine_label.setText("No medicine scanned.")
            return

        self.scan_status_label.setText(f"{user_name} recognized. Scan a medicine barcode.")
        self.medicine_label.setText("Waiting for a barcode assigned to this user.")

    def _scan_medicine_if_ready(self, bgr_frame):
        if self.active_user == UNKNOWN_USER:
            return

        self.medicine_registry.reload()
        barcode_results = self.barcode_scanner.scan(bgr_frame)
        if not barcode_results:
            return

        barcode_result = barcode_results[0]
        barcode_value = barcode_result["value"]
        scan_key = f"{self.active_user}|{barcode_value}"
        if scan_key == self.last_scan_key:
            return

        self.last_scan_key = scan_key
        record = self.medicine_registry.find_for_user(barcode_value, self.active_user)
        if record is not None:
            self.scan_status_label.setText(
                f"Barcode matched for {self.active_user}: {barcode_value} ({barcode_result['type']})"
            )
            self.medicine_label.setText(self.medicine_registry.format_record_summary(record))
            return

        owner_name = self.medicine_registry.find_owner_for_barcode(barcode_value)
        if owner_name is not None:
            self.scan_status_label.setText(
                f"Barcode {barcode_value} belongs to {owner_name}, not {self.active_user}."
            )
            self.medicine_label.setText("Medicine match rejected because the recognized user does not own it.")
            return

        self.scan_status_label.setText(f"Barcode {barcode_value} was not found in medicine_records.json.")
        self.medicine_label.setText("Add the barcode and medicine details to the local registry file.")

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
