# AI Surveillance System (Streamlit + YOLOv8 + DeepFace)

This is a professional AI surveillance system with a Streamlit dashboard for final year project level implementation.

## Key Features

- **Live Camera Monitoring** with OpenCV video processing
- **Weapon Detection** using YOLOv8 with 0.6+ confidence threshold
  - Detects: `knife`, `scissors`, `baseball bat`, `bottle`
  - Only weapon detection triggers database/email alerts
- **Face Recognition** with face_recognition library (0.5 tolerance)
  - Upload and manage known faces in the Face Manager
  - Automatic encoding reload after adding new faces
- **Emotion Analysis** with DeepFace
  - Detects angry emotion (display only)
- **Alert System**:
  - Weapon alerts → save to database, email, and logs
  - Other triggers (unknown person, angry, too many persons) → visual display only
- **Alert Persistence**:
  - Annotated images saved to `alerts/`
  - Event logs in `logs/log.txt`
  - SQLite database (`alerts.db`)
  - Email notifications with images (SMTP)
- **PDF Report Generation** from stored alerts

## Folder Structure

```text
ai-surveillance-system/
  app.py
  detection.py
  face_recognition_module.py
  database.py
  email_alert.py
  report.py
  requirements.txt
  README.md
  .gitignore
  yolov8n.pt               (optional; auto-download if missing)
  alerts.db                (auto-created at runtime)
  alerts/
  logs/
  faces/
```

## Setup

1. Create and activate a Python 3.10 virtual environment.
2. Install requirements:

```bash
pip install -r requirements.txt
```

### YOLO weights

The app uses `yolov8n.pt`. If it is not present in the project root, Ultralytics will attempt to download it automatically.

### Faces

Add known people photos into `faces/`. The filename (without extension) becomes the person name.
For example: `faces/alice.jpg` -> person name `alice`.

## SMTP Email Alerts

Email alerts require SMTP configuration. Set these environment variables before running the app:

- `SMTP_HOST`
- `SMTP_PORT` (usually `587` or `465`)
- `SMTP_USER`
- `SMTP_PASS`
- `SMTP_FROM`
- `SMTP_TO` (comma-separated or a single recipient)

The app also offers an Email settings panel in the UI.

## Run

From `ai-surveillance-system/`:

```bash
streamlit run app.py --server.port 8501
```

## Key Improvements

### 1. Fixed False Weapon Detection
- Added dedicated weapon confidence threshold of 0.6+ (higher than base YOLO confidence)
- Filters out low-confidence detections to prevent false positives
- Only triggers alerts for high-confidence weapon detections

### 2. Improved Face Recognition
- Uses `large` model for better accuracy
- Proper encoding reload when new faces are added
- Better error handling for edge cases
- Filters out `.gitkeep` files

### 3. Alert Logic Refinement
- **Weapon Detection**: Saves to database, logs, emails (PRIMARY ALERT)
- **Other Triggers**: Display-only visual indicators (no database/email spam)
- Clear UI labels to distinguish alert types
- Cooldown periods prevent alert flooding

### 4. Better Stability
- Try-catch blocks around processing pipeline
- Minimum face size check (48px) for emotion detection
- Graceful error recovery in camera loop
- Proper camera resource cleanup

### 5. Enhanced UI/UX
- Clear indication of which triggers save alerts vs display only
- Face manager properly invalidates caches when adding faces
- Better status messages (weapon alerts vs monitoring)
- Detection settings documented in sidebar

## Usage Tips

1. **Add Known Faces**: Use Face Manager to upload photos of people to recognize
2. **Camera Setup**: Ensure your camera is accessible (index 0 is default)
3. **Weapon Detection**: Position camera to clearly see objects; high confidence required
4. **Monitoring**: Green = known person, Red = unknown/angry/weapon
5. **Alerts**: Check Database/Images/Logs tabs to review saved weapon alerts

## Testing Notes

Camera streaming uses an OpenCV loop; emotion detection can be computationally expensive. For best results, keep the number of faces/emotions per frame reasonable and ensure your camera is accessible.

Weapon detection requires clear visibility and proper lighting for accurate classification.

