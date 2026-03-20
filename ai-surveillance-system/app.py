from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Optional

import cv2
import pandas as pd
import streamlit as st

from database import AlertDatabase
from detection import SurveillanceProcessor
from email_alert import EmailAlerter, SMTPConfig
from face_recognition_module import FaceRecognitionModule
from report import PDFReportGenerator


BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / "faces"
ALERTS_DIR = BASE_DIR / "alerts"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "alerts.db"
YOLO_WEIGHTS_PATH = BASE_DIR / "yolov8n.pt"


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


@st.cache_resource
def get_database() -> AlertDatabase:
    return AlertDatabase(DB_PATH)


@st.cache_resource
def get_face_module(faces_version: int) -> FaceRecognitionModule:
    # faces_version changes when the user uploads a new face.
    return FaceRecognitionModule(FACES_DIR, backend="auto")


@st.cache_resource
def get_email_alerter(
    *,
    enabled: bool,
    host: str,
    port: int,
    user: str,
    password: str,
    sender: str,
    recipients_csv: str,
) -> Optional[EmailAlerter]:
    if not enabled:
        return None
    recipients = [r.strip() for r in recipients_csv.split(",") if r.strip()]
    if not recipients:
        return None

    smtp_cfg = SMTPConfig(
        host=host,
        port=int(port),
        user=user,
        password=password,
        sender=sender,
        recipients=recipients,
        use_tls=True,
    )
    return EmailAlerter(smtp_cfg)


@st.cache_resource
def get_processor(
    *,
    yolo_conf: float,
    angry_threshold: float,
    emotion_every_n_frames: int,
    email_enabled: bool,
    email_host: str,
    email_port: int,
    email_user: str,
    email_password: str,
    email_sender: str,
    email_recipients_csv: str,
    faces_version: int,
) -> SurveillanceProcessor:
    db = get_database()
    face_module = get_face_module(faces_version)
    email_alerter = get_email_alerter(
        enabled=email_enabled,
        host=email_host,
        port=email_port,
        user=email_user,
        password=email_password,
        sender=email_sender,
        recipients_csv=email_recipients_csv,
    )

    # Ensure directories exist.
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FACES_DIR.mkdir(parents=True, exist_ok=True)

    # If weights file is missing, Ultralytics will attempt download when constructing YOLO.
    return SurveillanceProcessor(
        yolo_weights_path=YOLO_WEIGHTS_PATH,
        face_module=face_module,
        db=db,
        email_alerter=email_alerter,
        alerts_dir=ALERTS_DIR,
        logs_dir=LOGS_DIR,
        yolo_conf=yolo_conf,
        emotion_angry_threshold=angry_threshold,
        emotion_every_n_frames=emotion_every_n_frames,
    )


def _render_dashboard(db: AlertDatabase) -> None:
    st.subheader("Overview")

    total_alerts = db.count_alerts()
    last = db.fetch_last_alert()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total alerts", total_alerts)
    with c2:
        st.metric("Last alert", last.time_iso if last else "None")

    st.divider()

    st.subheader("Recent Alerts")
    events = db.fetch_alerts(limit=25, offset=0)
    if events:
        df = pd.DataFrame([{"time": e.time_iso, "type": e.type, "name": e.name} for e in events])
        st.dataframe(df, use_container_width=True, height=360)
    else:
        st.info("No alerts yet. Start the Live Camera to generate events.")

    st.divider()
    st.subheader("Alert Images")

    # Show the newest images.
    events_for_images = db.fetch_alerts(limit=12, offset=0)
    cols = st.columns(3)
    col_idx = 0
    for e in events_for_images:
        img_path = Path(e.image)
        if not img_path.exists():
            continue
        with cols[col_idx % 3]:
            st.image(str(img_path), use_container_width=True, caption=f"{e.type}: {e.name}")
        col_idx += 1


def _render_live_camera(processor: SurveillanceProcessor) -> None:
    st.subheader("Live Camera")

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if "last_triggered" not in st.session_state:
        st.session_state.last_triggered = ""

    camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=4, value=0)

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Start", type="primary"):
            st.session_state.camera_running = True
    with colB:
        if st.button("Stop"):
            st.session_state.camera_running = False
    with colC:
        st.write(" ")

    triggers = st.sidebar.container()
    with triggers:
        st.markdown("### Alert Triggers")
        weapon_trigger = st.checkbox("Weapon detected", value=True)
        emotion_trigger = st.checkbox("Angry emotion", value=True)
        unknown_person_trigger = st.checkbox("Unknown person", value=True)
        too_many_persons_trigger = st.checkbox("More than 2 persons", value=True)

        too_many_persons_threshold = st.slider("Persons threshold", min_value=2, max_value=10, value=2)

    status = st.empty()
    frame_slot = st.empty()

    if not st.session_state.camera_running:
        st.info("Click Start to begin the live feed.")
        return

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        st.error("Could not open camera. Check camera index and permissions.")
        st.session_state.camera_running = False
        return

    try:
        while st.session_state.camera_running:
            ok, frame = cap.read()
            if not ok or frame is None:
                status.warning("Failed to read frame.")
                time.sleep(0.2)
                continue

            annotated, triggered_types = processor.process_frame(
                frame,
                unknown_person_trigger=unknown_person_trigger,
                emotion_trigger=emotion_trigger,
                weapon_trigger=weapon_trigger,
                too_many_persons_trigger=too_many_persons_trigger,
                too_many_persons_threshold=int(too_many_persons_threshold),
            )

            frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            if triggered_types:
                # Show only the last event type to reduce UI spam.
                st.session_state.last_triggered = triggered_types[-1]
                status.error(f"Alert triggered: {', '.join(triggered_types)}")
            else:
                status.info("No alert in this frame.")

            # Small sleep to avoid maxing CPU.
            time.sleep(0.03)
    finally:
        cap.release()
        st.session_state.camera_running = False


def _render_face_manager(faces_version: int) -> int:
    st.subheader("Face Manager")

    ALERT_HELP = "Add known faces. The filename without extension becomes the person name."
    st.caption(ALERT_HELP)

    st.write("Existing faces:")
    face_files = sorted(
        [p for p in FACES_DIR.iterdir() if p.is_file() and p.name != ".gitkeep"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if face_files:
        cols = st.columns(4)
        for i, p in enumerate(face_files):
            with cols[i % 4]:
                st.image(str(p), use_container_width=True, caption=p.stem)
    else:
        st.info("No known faces yet. Upload one below.")

    st.divider()

    uploaded = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png", "webp"])
    person_name = st.text_input("Enter person name (filename will be this name)", value="")
    ext = Path(uploaded.name).suffix.lower() if uploaded else ".jpg"

    if st.button("Save face"):
        if uploaded is None:
            st.warning("Please upload an image first.")
            return faces_version
        person_name = person_name.strip()
        if not person_name:
            st.warning("Please enter a person name.")
            return faces_version

        out_path = FACES_DIR / f"{person_name}{ext}"
        with out_path.open("wb") as f:
            f.write(uploaded.getbuffer())

        st.success(f"Saved: {out_path.name}")

        # Reload encodings immediately so recognition can find the new face.
        face_module = get_face_module(faces_version)
        face_module.load_faces()

        return faces_version

    return faces_version


def _render_database(db: AlertDatabase) -> None:
    st.subheader("Database")
    limit = st.slider("Limit rows", min_value=10, max_value=500, value=100, step=10)

    events = db.fetch_alerts(limit=limit, offset=0)
    if not events:
        st.info("No alerts in database yet.")
        return

    df = pd.DataFrame([{"time": e.time_iso, "type": e.type, "image": e.image, "name": e.name} for e in events])
    st.dataframe(df, use_container_width=True, height=520)


def _render_images() -> None:
    st.subheader("Images")
    st.caption("Saved alert images from `alerts/`.")

    image_files = sorted(
        [p for p in ALERTS_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not image_files:
        st.info("No alert images yet.")
        return

    cols = st.columns(3)
    for i, p in enumerate(image_files):
        with cols[i % 3]:
            st.image(str(p), use_container_width=True)


def _render_logs() -> None:
    st.subheader("Logs")
    log_text = _safe_read_text(LOGS_DIR / "log.txt")
    if not log_text.strip():
        st.info("No logs yet.")
        return

    st.text_area("Event log", value=log_text, height=500)


def _render_report(db: AlertDatabase) -> None:
    st.subheader("PDF Report")

    limit = st.slider("Number of recent alerts to include", min_value=10, max_value=500, value=200, step=10)
    output_pdf = BASE_DIR / "report.pdf"

    if st.button("Generate report"):
        gen = PDFReportGenerator(db)
        pdf_path = gen.generate(output_pdf_path=output_pdf, limit=int(limit))
        st.success(f"Generated: {pdf_path.name}")

    if output_pdf.exists():
        st.download_button(
            label="Download report.pdf",
            data=output_pdf.read_bytes(),
            file_name="report.pdf",
            mime="application/pdf",
        )
    else:
        st.info("Generate a report to enable download.")


def main() -> None:
    st.set_page_config(page_title="AI Surveillance System", layout="wide")

    # Shared state
    if "faces_version" not in st.session_state:
        st.session_state.faces_version = 0
    if "email_enabled" not in st.session_state:
        st.session_state.email_enabled = False

    # Sidebar
    st.sidebar.title("AI Surveillance")
    page = st.sidebar.radio(
        "Menu",
        ["Dashboard", "Live Camera", "Face Manager", "Database", "Images", "Logs", "Report"],
        index=0,
    )

    # Global detection settings
    st.sidebar.markdown("### Detection Settings")
    yolo_conf = st.sidebar.slider("YOLO confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
    angry_threshold = st.sidebar.slider("Angry threshold", min_value=0.05, max_value=1.0, value=0.5, step=0.05)
    emotion_every_n_frames = st.sidebar.slider("Emotion every N frames", min_value=1, max_value=10, value=3, step=1)

    # Email settings
    st.sidebar.markdown("### Email Alerts (SMTP)")
    email_enabled = st.sidebar.checkbox("Enable email alerts", value=bool(os.environ.get("SMTP_HOST")))

    email_host = st.sidebar.text_input("SMTP_HOST", value=os.environ.get("SMTP_HOST", ""))
    email_port = st.sidebar.number_input("SMTP_PORT", min_value=1, max_value=65535, value=int(os.environ.get("SMTP_PORT", "587") or "587"))
    email_user = st.sidebar.text_input("SMTP_USER", value=os.environ.get("SMTP_USER", ""), disabled=not email_enabled)
    email_password = st.sidebar.text_input("SMTP_PASS", value=os.environ.get("SMTP_PASS", ""), type="password", disabled=not email_enabled)
    email_sender = st.sidebar.text_input("SMTP_FROM", value=os.environ.get("SMTP_FROM", ""), disabled=not email_enabled)
    email_recipients_csv = st.sidebar.text_input("SMTP_TO (comma-separated)", value=os.environ.get("SMTP_TO", ""), disabled=not email_enabled)

    db = get_database()

    # Build/cached processor with the current settings + faces.
    processor = get_processor(
        yolo_conf=float(yolo_conf),
        angry_threshold=float(angry_threshold),
        emotion_every_n_frames=int(emotion_every_n_frames),
        email_enabled=bool(email_enabled),
        email_host=str(email_host),
        email_port=int(email_port),
        email_user=str(email_user),
        email_password=str(email_password),
        email_sender=str(email_sender),
        email_recipients_csv=str(email_recipients_csv),
        faces_version=int(st.session_state.faces_version),
    )

    if page == "Dashboard":
        _render_dashboard(db)
    elif page == "Live Camera":
        _render_live_camera(processor)
    elif page == "Face Manager":
        st.session_state.faces_version = _render_face_manager(st.session_state.faces_version)
    elif page == "Database":
        _render_database(db)
    elif page == "Images":
        _render_images()
    elif page == "Logs":
        _render_logs()
    elif page == "Report":
        _render_report(db)


if __name__ == "__main__":
    main()

