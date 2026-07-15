# Face Recognition Project

This project uses `PySide6` and `OpenCV` to show a live camera feed, recognize a saved user by face, and guide the user to scan a medicine barcode after recognition.

## Current Behavior

- A large centered guide box is shown on the camera preview.
- The guide box is `red` when the face is outside the target area.
- The guide box turns `green` when the detected face is fully inside the target area.
- If the face matches a saved user, the recognition logic stays the same and the overlay changes to `blue` to tell the user they can now scan the medicine.
- After the user is recognized, detected barcode outlines are shown on screen.
- The barcode outline stays `red` when it is outside the target box.
- The barcode outline turns `blue` when the barcode is fully inside the target box.
- The label under the camera shows the recognized user name or `Unknown User`.
- When a scanned barcode belongs to the recognized user, the app shows the saved medicine details.
- When a scanned barcode exists but belongs to another user, the app reports that ownership mismatch.
- When a scanned barcode does not exist in the local records, the app reports that no medicine record was found for the recognized user.
- If the current face is aligned and not already saved, you can use `Save Current User` to register that person.

Saved user face images are stored in `user_images/`.
Medicine records are stored in `medicine_records.json`.

## Project Files

- `main.py`: Starts the GUI and updates the live camera frame.
- `face_alignment.py`: Draws the face target box, face box, and barcode/object overlay colors.
- `user_recognition.py`: Loads saved users, detects faces, and matches them.
- `barcode_scanner.py`: Detects barcodes and returns their outline points.
- `medicine_registry.py`: Loads medicine/barcode records from `medicine_records.json`.
- `medicine_records.json`: Local barcode-to-user medicine records used by the app.

## How To Run

Run the app from the project root folder:

```powershell
.\.venv\Scripts\python.exe main.py
```

If the virtual environment is not active yet, that command is enough in this project because it directly uses the local Python inside `.venv`.

## Requirements

- Windows
- A working webcam
- The project virtual environment in `.venv`
- Python packages used by the app, including `PySide6` and `opencv-python`

## Basic Use

1. Start the application.
2. Click `Start Camera`.
3. Move your face into the centered box until it turns green.
4. If your face is recognized, the box turns blue.
5. Hold the medicine barcode inside the same box.
6. When the barcode outline turns blue, it is inside the scan area.
7. If the barcode matches the recognized user, the medicine details appear below the camera feed.
8. If the barcode belongs to another saved user, the app reports that the barcode belongs to someone else.
9. To register a new person, keep the face aligned and click `Save Current User`.
