from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from database import AlertDatabase
from email_alert import EmailAlerter
from face_recognition_module import FaceRecognitionModule, FaceMatch


@dataclass(frozen=True)
class WeaponDetection:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]  # x1,y1,x2,y2


class YoloDetector:
    def __init__(self, weights_path: str | Path, *, conf: float = 0.35, iou: float = 0.5):
        self.weights_path = Path(weights_path)
        self.conf = conf
        self.iou = iou

        # Lazy import to keep Streamlit startup smoother.
        from ultralytics import YOLO

        # If a weights file is missing, fall back to the basename (e.g. `yolov8n.pt`)
        # so Ultralytics can download by model name.
        model_input = str(self.weights_path) if self.weights_path.exists() else self.weights_path.name
        self.model = YOLO(model_input)
        self.names = self.model.names

    def detect(self, frame_bgr: np.ndarray):
        results = self.model.predict(source=frame_bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
        return results


class EmotionDetector:
    def __init__(self, *, angry_threshold: float = 0.5):
        self.angry_threshold = angry_threshold

        # Import lazily for speed/robustness.
        try:
            from deepface import DeepFace  # noqa: F401

            self._deepface_ok = True
        except Exception:
            self._deepface_ok = False

    def is_angry(self, emotion_scores: dict[str, float]) -> tuple[bool, float]:
        angry_score = float(emotion_scores.get("angry", 0.0))
        is_angry = angry_score >= self.angry_threshold
        return is_angry, angry_score

    def analyze_faces(self, face_crops_bgr: list[np.ndarray]) -> list[tuple[bool, float]]:
        if not self._deepface_ok:
            return [(False, 0.0) for _ in face_crops_bgr]

        from deepface import DeepFace

        outputs: list[tuple[bool, float]] = []
        for crop in face_crops_bgr:
            if crop is None or crop.size == 0:
                outputs.append((False, 0.0))
                continue

            try:
                # DeepFace can accept ndarray inputs.
                result = DeepFace.analyze(
                    crop,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True,
                )
                # DeepFace may return dict or list[dict]
                if isinstance(result, list) and result:
                    result = result[0]

                emotion_scores = result.get("emotion", {}) if isinstance(result, dict) else {}
                outputs.append(self.is_angry(emotion_scores))
            except Exception:
                outputs.append((False, 0.0))

        return outputs


class AlertManager:
    def __init__(
        self,
        *,
        alerts_dir: str | Path,
        logs_dir: str | Path,
        db: AlertDatabase,
        email_alerter: Optional[EmailAlerter] = None,
        logs_log_txt_name: str = "log.txt",
        cooldown_seconds: Optional[dict[str, int]] = None,
    ):
        self.alerts_dir = Path(alerts_dir)
        self.logs_dir = Path(logs_dir)
        self.db = db
        self.email_alerter = email_alerter

        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.logs_dir / logs_log_txt_name

        if cooldown_seconds is None:
            cooldown_seconds = {
                "weapon_detected": 20,
                "angry_emotion": 20,
                "unknown_person": 20,
                "too_many_persons": 20,
            }
        self.cooldown_seconds = cooldown_seconds

        self._last_alert_ts: dict[str, float] = {}

        # Ensure log file exists.
        if not self.log_path.exists():
            self.log_path.write_text("", encoding="utf-8")

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    def _should_trigger(self, event_type: str) -> bool:
        now = time.time()
        last = self._last_alert_ts.get(event_type, 0.0)
        wait = self.cooldown_seconds.get(event_type, 20)
        return (now - last) >= wait

    def _append_log(self, *, event_type: str, name: str, image_path: Path) -> None:
        line = f"[{self._now_iso()}] type={event_type} name={name} image={image_path.name}\n"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def trigger_alert(self, *, event_type: str, frame_bgr: np.ndarray, name: str) -> Optional[Path]:
        if not self._should_trigger(event_type):
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_filename = f"{event_type}_{ts}.jpg"
        image_path = self.alerts_dir / image_filename

        # Save annotated frame
        ok = cv2.imwrite(str(image_path), frame_bgr)
        if not ok:
            return None

        # DB + log
        self.db.insert_alert(event_type, image_path=str(image_path), name=name)
        self._append_log(event_type=event_type, name=name, image_path=image_path)

        # Email in background to avoid blocking camera loop.
        if self.email_alerter is not None:
            subject = f"AI Alert: {event_type} ({name})"
            body = f"An alert was triggered.\nTime: {self._now_iso()}\nType: {event_type}\nName: {name}\nImage: {image_path.name}"
            img_path_copy = image_path

            threading.Thread(
                target=lambda: self.email_alerter.send_alert(subject=subject, body=body, image_path=img_path_copy),
                daemon=True,
            ).start()

        self._last_alert_ts[event_type] = time.time()
        return image_path


class SurveillanceProcessor:
    WEAPONS = {"scissors", "knife", "bottle", "baseball bat"}

    def __init__(
        self,
        *,
        yolo_weights_path: str | Path,
        face_module: FaceRecognitionModule,
        db: AlertDatabase,
        email_alerter: Optional[EmailAlerter] = None,
        alerts_dir: str | Path,
        logs_dir: str | Path,
        yolo_conf: float = 0.35,
        yolo_iou: float = 0.5,
        emotion_angry_threshold: float = 0.5,
        emotion_every_n_frames: int = 3,
    ):
        self.yolo = YoloDetector(yolo_weights_path, conf=yolo_conf, iou=yolo_iou)
        self.face_module = face_module
        self.db = db

        self.emotion_detector = EmotionDetector(angry_threshold=emotion_angry_threshold)
        self.alert_manager = AlertManager(
            alerts_dir=alerts_dir,
            logs_dir=logs_dir,
            db=db,
            email_alerter=email_alerter,
        )
        self.emotion_every_n_frames = max(1, int(emotion_every_n_frames))

        self._frame_idx = 0

    @staticmethod
    def _draw_label(img: np.ndarray, text: str, x: int, y: int, *, color: tuple[int, int, int]):
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x, y - th - baseline), (x + tw + 4, y + 4), color, -1)
        cv2.putText(img, text, (x + 2, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    @staticmethod
    def _draw_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, *, color: tuple[int, int, int], thickness: int = 2):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    def _extract_yolo_detections(self, results) -> tuple[list[tuple[int, int, int, int]], list[WeaponDetection]]:
        person_boxes: list[tuple[int, int, int, int]] = []
        weapon_dets: list[WeaponDetection] = []

        if results is None or results.boxes is None:
            return person_boxes, weapon_dets

        boxes = results.boxes
        if boxes.xyxy is None or boxes.cls is None or boxes.conf is None:
            return person_boxes, weapon_dets

        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
            label = str(self.yolo.names.get(int(cls_id), str(cls_id))).strip().lower()
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            if label == "person":
                person_boxes.append((x1i, y1i, x2i, y2i))
            if label in self.WEAPONS:
                weapon_dets.append(
                    WeaponDetection(
                        label=str(label),
                        confidence=float(conf),
                        bbox_xyxy=(x1i, y1i, x2i, y2i),
                    )
                )

        return person_boxes, weapon_dets

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        unknown_person_trigger: bool = True,
        emotion_trigger: bool = True,
        weapon_trigger: bool = True,
        too_many_persons_trigger: bool = True,
        too_many_persons_threshold: int = 2,
        save_annotated_on_alert: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Returns:
          - annotated frame
          - list of triggered event types in this frame (can be empty)
        """

        self._frame_idx += 1
        annotated = frame_bgr.copy()

        results = self.yolo.detect(frame_bgr)
        person_boxes, weapon_dets = self._extract_yolo_detections(results)

        # Face recognition (names + face boxes)
        faces: list[FaceMatch] = self.face_module.recognize(frame_bgr)

        # Emotion analysis only periodically
        angry_flags: list[bool] = [False] * len(faces)
        angry_scores: list[float] = [0.0] * len(faces)
        should_analyze_emotions = (self._frame_idx % self.emotion_every_n_frames == 0)
        if should_analyze_emotions and faces:
            face_crops: list[np.ndarray] = []
            for f in faces:
                top, right, bottom, left = f.box
                top = max(0, top)
                left = max(0, left)
                bottom = min(frame_bgr.shape[0], bottom)
                right = min(frame_bgr.shape[1], right)
                crop = frame_bgr[top:bottom, left:right]
                face_crops.append(crop)

            emotion_results = self.emotion_detector.analyze_faces(face_crops)
            for i, (is_angry, score) in enumerate(emotion_results):
                angry_flags[i] = bool(is_angry)
                angry_scores[i] = float(score)

        # Draw YOLO objects
        for (x1, y1, x2, y2) in person_boxes:
            self._draw_box(annotated, x1, y1, x2, y2, color=(0, 255, 255), thickness=2)
            self._draw_label(annotated, "PERSON", x1, max(18, y1), color=(0, 255, 255))

        # Weapon alerts are weapon-only.
        weapon_detected = False
        for w in weapon_dets:
            weapon_detected = True  # set True only when YOLO detects a weapon class
            x1, y1, x2, y2 = w.bbox_xyxy
            self._draw_box(annotated, x1, y1, x2, y2, color=(0, 0, 255), thickness=2)
            self._draw_label(
                annotated,
                f"{w.label} {w.confidence:.2f}",
                x1,
                max(18, y1),
                color=(0, 0, 255),
            )

        # Draw faces
        unknown_present = False
        angry_present = False
        unknown_names: list[str] = []

        for idx, face in enumerate(faces):
            top, right, bottom, left = face.box
            x1, y1, x2, y2 = left, top, right, bottom
            name = face.name
            is_angry = angry_flags[idx] if idx < len(angry_flags) else False

            if name == "Unknown":
                unknown_present = True
                unknown_names.append(name)

            if is_angry:
                angry_present = True

            color = (0, 255, 0)  # green default
            label_prefix = name
            if is_angry:
                color = (0, 0, 255)
                label_prefix = f"{name} (ANGRY)"
            elif name == "Unknown":
                color = (0, 0, 255)

            self._draw_box(annotated, x1, y1, x2, y2, color=color, thickness=2)
            self._draw_label(annotated, label_prefix, x1, max(18, y1), color=color)

        weapons_joined = ", ".join(sorted({w.label for w in weapon_dets})) if weapon_dets else ""

        triggered_types: list[str] = []
        # Alerts are weapon-only. Other conditions may be displayed on-screen, but should not be persisted.

        # Trigger alert ONLY when a weapon is detected by YOLO.
        if weapon_trigger and weapon_detected == True:
            triggered = self.alert_manager.trigger_alert(
                event_type="weapon_detected",
                frame_bgr=annotated if save_annotated_on_alert else frame_bgr,
                name=weapons_joined,
            )
            if triggered is not None:
                triggered_types.append("weapon_detected")

        return annotated, triggered_types

