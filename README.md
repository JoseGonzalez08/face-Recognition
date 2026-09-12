# Face Recognition Medicine Barcode Prototype

This project is a Windows desktop prototype built with Python, PySide6, and OpenCV. It demonstrates a local workflow that recognizes a saved user, opens a five-minute barcode-scanning session, accepts a barcode only when it is fully inside the camera guide, looks up the decoded value in a local medicine registry, and displays a color-coded test result.

The project is intended for software testing and learning. All included medicine details, symptoms, warnings, and allergy reactions are simulated test data. This application does not provide medical advice and must not be used to make real medication decisions.

## Prototype Workflow

The application follows this sequence:

```text
Camera stopped
    ↓
Start camera
    ↓
Detect and recognize a saved face
    ↓
Lock the recognized user for a five-minute prototype session
    ↓
Detect and decode barcodes
    ↓
Require the complete barcode outline to be inside the guide box
    ↓
Look up the decoded value in medicine_records.json
    ↓
Display and log the result
    ↓
Stop the camera and clear the user when five minutes expire
```

Recognition remains locked to the first recognized user for the duration of the session. The session ends when the user clicks `Stop Camera`, closes the application, or reaches the five-minute limit. When the limit expires, the application stops the camera but leaves the window open so another test session can begin.

## Camera Guide Colors

The camera overlay and the medicine result use related but separate colors.

### Face and barcode overlay

- **Red guide:** The face is missing or outside the target area.
- **Green guide:** A detected face is fully inside the target area but has not been recognized yet.
- **Blue guide:** A saved user has been recognized and barcode-scanning mode is active.
- **Red barcode outline:** A barcode was detected outside the target area.
- **Blue barcode outline:** The complete barcode is inside the target and can be accepted.

The application checks every detected corner of the barcode. A decoded value is not looked up until all corners are inside the target box.

### Lookup result colors

- **Green:** The barcode exists, belongs to the recognized user, and its test record has a green status.
- **Yellow:** The barcode belongs to another user, or the matched test record contains simulated symptoms that should be reviewed.
- **Red:** The decoded barcode does not exist in the registry, or the matched test record contains a simulated allergy/blocking warning.
- **Orange instruction:** A barcode was detected outside the scan box and was not accepted.

An image that cannot be decoded produces no barcode result because the passive camera scanner cannot distinguish an unreadable attempt from having no barcode in view. The red unknown result therefore means that OpenCV decoded the barcode successfully, but its value was not found in the local registry.

## Main Project Files

- `main.py` creates the PySide6 window, manages the camera and five-minute timer, locks the recognized user, filters barcode results by position, selects the result color, updates the labels, and prevents duplicate log entries while one barcode remains visible.
- `user_recognition.py` loads saved face images, detects the largest face, converts it into a normalized grayscale template, and compares it with the saved templates.
- `face_alignment.py` calculates the target box, detects the largest visible face, checks whether a barcode polygon is completely inside the target, and draws the colored overlays.
- `barcode_scanner.py` wraps OpenCV's `barcode_BarcodeDetector`, returning the decoded value, barcode type, and corner coordinates.
- `medicine_registry.py` loads local JSON records into `MedicineRecord` objects, matches a barcode and user, finds a barcode's owner, and formats the result shown in the interface.
- `medicine_records.json` contains the local prototype users, medicines, statuses, symptoms, and warnings.
- `scan_logger.py` appends accepted scan events to the local `scan_events.jsonl` file.
- `test_barcodes/generate_test_barcode.py` validates UPC-A check digits and generates all five reproducible test cards.
- `tests/test_barcode_workflow.py` tests target-box geometry, medicine outcomes, event logging, UPC-A generation, and decoding with the application's real scanner.
- `requirements.txt` records the Python package versions used by this prototype.
- `run.bat` starts the application using the local virtual environment.

## Installation

The current project is designed for Windows with a working webcam and Python installed.

From PowerShell in the project directory, create the virtual environment if it does not already exist:

```powershell
python -m venv .venv
```

Install the dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the application:

```powershell
.\.venv\Scripts\python.exe main.py
```

You can also start it with:

```powershell
.\run.bat
```

## Using the Application

1. Click `Start Camera`.
2. Place a visible face inside the guide box.
3. Wait for the saved user to be recognized. The guide changes to blue and the five-minute timer begins.
4. Show a barcode to the camera.
5. Move the entire barcode inside the guide. A red barcode outline means it is outside; a blue outline means it is accepted.
6. Read the color-coded prototype result below the camera image.
7. Remove the barcode before scanning it again. Keeping the same barcode continuously visible creates only one log event.
8. Click `Stop Camera` to end early, or wait for the five-minute session timeout.

To register a local test user, start the camera, keep one face clearly visible, click `Save Current User`, and enter a name. The cropped grayscale face image is written to `user_images/` and ignored by Git.

## Medicine Registry Format

Records in `medicine_records.json` use this structure:

```json
{
  "barcode": "987654321098",
  "user_name": "Jose",
  "medicine_name": "Demo Relief Tablet 25 mg",
  "description": "A fictional medicine used to demonstrate a symptom caution.",
  "uses": "Prototype testing only.",
  "directions": "Review the simulated effects shown by the prototype.",
  "demo_notice": "All details are invented for software testing.",
  "demo_status": "yellow",
  "demo_symptoms": ["mild dizziness", "dry mouth", "sleepiness"],
  "demo_warning": "The test profile reports simulated symptoms that should be reviewed."
}
```

`demo_status` should be `green`, `yellow`, or `red`. An unsupported status is displayed as yellow by the interface. User-name comparisons ignore capitalization and repeated spaces, while barcode values must match exactly.

## Scan Event Log

Accepted scans are appended as JSON Lines records in `scan_events.jsonl`. The file is deliberately ignored by Git. Each line resembles:

```json
{"timestamp_utc":"2026-09-12T22:00:00+00:00","user_name":"Jose","barcode":"987654321098","barcode_type":"UPC_A","outcome":"matched_yellow"}
```

Logged fields are:

- UTC timestamp
- Recognized user name
- Decoded barcode value
- Barcode format reported by OpenCV
- Outcome such as `matched_green`, `matched_yellow`, `matched_red`, `wrong_user`, or `unknown`

Camera frames and face images are not written to the scan log.

## Barcode Test Scenarios

The generated PNG cards are in `test_barcodes/`. Open one on a phone or second monitor, recognize the saved Jose test user, and place the complete card inside the camera guide.

| Card | UPC-A value | Expected result |
| --- | --- | --- |
| [`01_green_known_medicine.png`](test_barcodes/01_green_known_medicine.png) | `036000291452` | Green: known medicine assigned to Jose |
| [`02_yellow_wrong_user.png`](test_barcodes/02_yellow_wrong_user.png) | `135791357917` | Yellow: decoded medicine belongs to fictional user Alex |
| [`03_red_unknown_barcode.png`](test_barcodes/03_red_unknown_barcode.png) | `246802468024` | Red: decoded value is absent from the registry |
| [`04_yellow_simulated_symptoms.png`](test_barcodes/04_yellow_simulated_symptoms.png) | `987654321098` | Yellow: known medicine with invented dizziness, dry mouth, and sleepiness |
| [`05_red_simulated_allergy.png`](test_barcodes/05_red_simulated_allergy.png) | `314159265358` | Red: known medicine with an invented prior allergic reaction |

Regenerate all cards with:

```powershell
.\.venv\Scripts\python.exe test_barcodes\generate_test_barcode.py
```

The generator validates each UPC-A check digit before creating an image. It adds slight blur and resizing to approximate a camera image because OpenCV may not decode perfectly sharp synthetic bars reliably.

## Automated Verification

Run the test suite from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests verify:

- A barcode fully inside the target is accepted.
- A barcode crossing a target edge is rejected.
- Green, yellow, red, wrong-user, and unknown registry records are classified correctly.
- Scan logs contain the expected fields.
- The sample UPC-A check digit and module count are correct.
- Every generated test card decodes through the same `BarcodeScanner` used by the application.

Automated tests do not replace testing with the actual webcam, lighting, focus, screen glare, viewing angle, and saved face image.

## Current Prototype Limitations

- Face identity is based on a simple grayscale pixel-difference comparison, not a production-quality face-recognition model.
- A recognized user remains locked for the entire five-minute session even if that person leaves the camera view.
- There is no liveness detection, password/PIN fallback, role system, or secure authentication.
- Face templates, medicine records, and scan logs are local files and are not encrypted.
- Camera processing runs on the GUI thread, so slow hardware may reduce responsiveness.
- Barcode reliability depends on lighting, focus, glare, distance, and the display or printed card.
- The registry is a manually maintained JSON test file rather than a validated medical database.
- The application does not verify prescriptions, dosage, drug interactions, allergies, or clinical instructions.
- The displayed medical-style content is fictional and exists only to exercise software states.

## Safety Notice

This repository is an educational prototype. Do not use its output to decide whether to start, stop, take, avoid, or change any real medication. Consult the medication label and an appropriately licensed healthcare professional for real medical decisions.
