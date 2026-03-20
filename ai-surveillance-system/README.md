# AI Surveillance System (Streamlit + YOLOv8 + DeepFace)

This is a modular AI surveillance system with a Streamlit dashboard:

- Live camera (OpenCV) + object detection (YOLOv8)
- Face recognition against images in `faces/`
- Emotion detection with DeepFace (angry detection)
- Alerts:
  - weapons detected (`scissors`, `knife`, `bottle`, `baseball bat`)
  - angry emotion detected
  - unknown person detected
  - more than 2 persons in frame
- Alert persistence:
  - save annotated images to `alerts/`
  - append to `logs/log.txt`
  - store events in SQLite (`alerts.db`)
  - send an email with the alert image (SMTP)
- PDF report generation from the SQLite database

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

## Testing Notes

Camera streaming uses an OpenCV loop; emotion detection can be computationally expensive. For best results, keep the number of faces/emotions per frame reasonable and ensure your camera is accessible.

