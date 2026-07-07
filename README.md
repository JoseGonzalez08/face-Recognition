# Face Recognition Project

Basic PySide6 and OpenCV project with webcam start and stop controls, saved
face recognition, and local medicine barcode matching.

Saved user face images are stored in `user_images/`. Start the camera, align
the face, then click `Save Current User` to register the current face with a
name. Once a saved face is recognized, the app starts looking for a medicine
barcode in the camera feed. If the barcode is assigned to the recognized user,
the app displays the medicine name, description, uses, and directions.

Medicine barcode assignments live in `medicine_records.json`. Each record is a
barcode tied to one user and one medicine. Update that file to add more users
or more medicines.

## Current workflow

1. Start the camera.
2. Let the app recognize a saved user face.
3. Once a known user is recognized, present a medicine barcode to the camera.
4. If the barcode belongs to that user, the app shows:
   - medicine name
   - assigned user name
   - description
   - uses
   - directions
5. If the barcode belongs to a different user or is missing from the registry,
   the app shows a rejection or not-found message.

## Local data files

- `user_images/`: saved face images used for recognition
- `medicine_records.json`: local barcode-to-user medicine registry

Example record shape:

```json
{
  "barcode": "036000291452",
  "user_name": "Jose",
  "medicine_name": "Acetaminophen 500 mg",
  "description": "Pain reliever and fever reducer tablets.",
  "uses": "Temporary relief of minor aches, pains, and fever.",
  "directions": "Take as directed on the package or by a licensed clinician."
}
```

## What has been set up

So far, the project uses a local Python virtual environment in `.venv/` and
runs the GUI from `main.py`. The original command to start the project was:

```powershell
.\.venv\Scripts\python.exe main.py
```

To make that easier, a Windows launcher script named `run.bat` was added. It
uses the Python executable inside `.venv/` and starts `main.py` for you.

## Run the project

From the root folder of the project, run:

```powershell
.\run
```

This is the shorter version of running the full virtual environment command.

## Status

Implemented:

- face recognition with saved user images
- face alignment box feedback
- local barcode scanning with OpenCV barcode detector
- local user-to-medicine matching
- on-screen medicine details after a valid match

Not implemented yet:

- live tested barcode scanner validation with real medicine packaging
- online medicine lookup from FDA / RxNorm
- UI for adding medicine records without editing JSON

