import sys

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QInputDialog, QVBoxLayout, QWidget

from barcode_scanner import BarcodeScanner
from face_alignment import annotate_face_alignment, get_target_box, polygon_inside_box
from medicine_registry import MedicineRegistry
from scan_logger import ScanEventLogger
from user_recognition import UNKNOWN_USER, UserRecognition


SESSION_TIMEOUT_MS = 5 * 60 * 1000


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
        self.scan_logger = ScanEventLogger()
        self.last_logged_scan = None
        self.active_user_name = UNKNOWN_USER
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.session_timer = QTimer()
        self.session_timer.setSingleShot(True)
        self.session_timer.timeout.connect(self.end_timed_out_session)

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
            self.last_logged_scan = None
        else:
            self.image_label.setText("Could not access camera")

    def stop_camera(self):
        self.timer.stop()
        self.session_timer.stop()
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
        self.last_logged_scan = None

    def end_timed_out_session(self):
        self.stop_camera()
        self.status_label.setText("Status: Session ended after the 5-minute prototype limit.")
        self.medicine_label.setText("Start the camera to begin a new session.")

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
                self.session_timer.start(SESSION_TIMEOUT_MS)

        user_name = self.active_user_name
        user_recognized = user_name != UNKNOWN_USER
        self.user_label.setText(f"User: {user_name}")
        self.user_label.setStyleSheet("color: green;" if user_recognized else "")

        barcode_results = self.barcode_scanner.scan(frame) if user_recognized else []
        barcode_points = [result["points"] for result in barcode_results if result["points"]]
        target_box = get_target_box(frame.shape[1], frame.shape[0])
        accepted_barcode_results = [
            result
            for result in barcode_results
            if result["points"] and polygon_inside_box(target_box, result["points"])
        ]
        barcode_detected_outside = bool(barcode_results) and not accepted_barcode_results
        self._update_status_and_medicine(
            user_name,
            accepted_barcode_results,
            barcode_detected_outside=barcode_detected_outside,
        )

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

    def _update_status_and_medicine(
        self, user_name, barcode_results, barcode_detected_outside=False
    ):
        if user_name == UNKNOWN_USER:
            self.status_label.setText("Status: Align your face inside the box for recognition.")
            self.status_label.setStyleSheet("")
            self.medicine_label.setText(
                "Medicine information will appear here after a barcode is scanned."
            )
            self.medicine_label.setStyleSheet("")
            self.last_logged_scan = None
            return

        self.status_label.setText(f"Status: {user_name} recognized. Scan the barcode.")
        self.status_label.setStyleSheet("color: green;")

        self.medicine_registry.reload()

        for barcode_result in barcode_results:
            record = self.medicine_registry.find_for_user(barcode_result["value"], user_name)
            if record is not None:
                self.medicine_label.setText(self.medicine_registry.format_record_summary(record))
                demo_status = record.demo_status if record.demo_status in {"green", "yellow", "red"} else "yellow"
                status_colors = {"green": "green", "yellow": "#b36b00", "red": "red"}
                self.medicine_label.setStyleSheet(f"color: {status_colors[demo_status]};")
                self.status_label.setText(
                    f"Status: Barcode accepted — {demo_status.upper()} demo result."
                )
                self.status_label.setStyleSheet(f"color: {status_colors[demo_status]};")
                self._log_scan_once(user_name, barcode_result, f"matched_{demo_status}")
                return

            owner_name = self.medicine_registry.find_owner_for_barcode(barcode_result["value"])
            if owner_name is not None:
                self.medicine_label.setText(
                    "YELLOW DEMO RESULT\n"
                    f"Barcode decoded, but it belongs to {owner_name}, not {user_name}."
                )
                self.medicine_label.setStyleSheet("color: #b36b00;")
                self.status_label.setText("Status: Barcode is not assigned to the recognized user.")
                self.status_label.setStyleSheet("color: #b36b00;")
                self._log_scan_once(user_name, barcode_result, "wrong_user")
                return

        if barcode_results:
            barcode_result = barcode_results[0]
            barcode_type = barcode_result["type"] or "Unknown format"
            self.medicine_label.setText(
                "RED DEMO RESULT — barcode not found in the local registry.\n"
                f"Value: {barcode_result['value']}\n"
                f"Format: {barcode_type}"
            )
            self.medicine_label.setStyleSheet("color: red;")
            self.status_label.setText("Status: Unrecognized medicine barcode.")
            self.status_label.setStyleSheet("color: red;")
            self._log_scan_once(user_name, barcode_result, "unknown")
            return

        self.last_logged_scan = None
        if barcode_detected_outside:
            self.status_label.setText("Status: Move the entire barcode inside the scan box.")
            self.status_label.setStyleSheet("color: #b36b00;")
            self.medicine_label.setText("Barcode detected outside the scan area; it was not accepted.")
            self.medicine_label.setStyleSheet("color: #b36b00;")
            return

        self.medicine_label.setText("Medicine information will appear here after a barcode is scanned.")
        self.medicine_label.setStyleSheet("")

    def _log_scan_once(self, user_name, barcode_result, outcome):
        scan_key = (user_name, barcode_result["value"], outcome)
        if scan_key == self.last_logged_scan:
            return

        self.scan_logger.log(user_name, barcode_result, outcome)
        self.last_logged_scan = scan_key

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
