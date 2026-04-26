# Face Recognition Project

Basic PySide6 and OpenCV project with webcam start and stop controls.

Saved user face images are stored in `user_images/`. Start the camera, align
the face, then click `Save Current User` to register the current face with a
name. The app displays the recognized user below the camera image.

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

