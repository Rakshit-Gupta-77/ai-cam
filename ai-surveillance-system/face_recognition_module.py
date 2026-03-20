from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceMatch:
    name: str
    box: tuple[int, int, int, int]  # top, right, bottom, left
    confidence: float


class FaceRecognitionModule:
    """
    Recognize faces on frames by comparing face embeddings against images in `faces/`.

    - Known people: files in `faces/` folder; filename stem becomes the person name.
    - Unknown faces: return name "Unknown".
    """

    def __init__(
        self,
        faces_dir: str | Path,
        *,
        backend: str = "face_recognition",
        face_match_threshold: float = 0.5,
        deepface_model_name: str = "Facenet512",
    ):
        self.faces_dir = Path(faces_dir)
        self.faces_dir.mkdir(parents=True, exist_ok=True)

        # Per requirements, we use `face_recognition` for matching.
        self.backend = backend
        self.face_match_threshold = float(face_match_threshold)  # used as `tolerance`
        self.deepface_model_name = deepface_model_name  # kept for compatibility (not used)

        import face_recognition  # noqa: F401

        self.known_names: list[str] = []
        self.known_encodings: list[np.ndarray] = []
        self.load_faces()

    def load_faces(self) -> int:
        """
        Reload known faces from `faces/` folder into in-memory encodings.

        Filename stem becomes the `person name`.

        Returns:
            Number of faces loaded successfully.
        """
        import face_recognition

        self.known_names.clear()
        self.known_encodings.clear()

        supported_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = [p for p in self.faces_dir.iterdir() if p.suffix.lower() in supported_exts and p.is_file() and p.name != ".gitkeep"]
        images.sort(key=lambda p: p.name.lower())

        loaded_count = 0
        for img_path in images:
            try:
                img = face_recognition.load_image_file(str(img_path))
                encodings = face_recognition.face_encodings(img, model="large")
                if not encodings:
                    continue
                self.known_names.append(img_path.stem)
                self.known_encodings.append(np.asarray(encodings[0], dtype=np.float32))
                loaded_count += 1
            except Exception:
                continue

        return loaded_count

    def recognize(self, frame_bgr: np.ndarray) -> list[FaceMatch]:
        """
        Return a list of recognized faces with bounding boxes and names.
        """

        if frame_bgr is None or frame_bgr.size == 0:
            return []

        import face_recognition

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model="hog")
            if not locations:
                return []

            encodings = face_recognition.face_encodings(rgb, known_face_locations=locations, model="large")

            results: list[FaceMatch] = []
            tolerance = float(self.face_match_threshold)

            for loc, encoding in zip(locations, encodings):
                if not self.known_encodings:
                    results.append(FaceMatch(name="Unknown", box=loc, confidence=0.0))
                    continue

                matches = face_recognition.compare_faces(
                    self.known_encodings,
                    encoding,
                    tolerance=tolerance,
                )

                if any(matches):
                    distances = face_recognition.face_distance(self.known_encodings, encoding)
                    best_idx = int(np.argmin(distances))
                    if matches[best_idx]:
                        name = self.known_names[best_idx]
                        dist = float(distances[best_idx])
                        confidence = float(max(0.0, min(1.0, 1.0 / (1.0 + dist))))
                        results.append(FaceMatch(name=name, box=loc, confidence=confidence))
                    else:
                        results.append(FaceMatch(name="Unknown", box=loc, confidence=0.0))
                else:
                    results.append(FaceMatch(name="Unknown", box=loc, confidence=0.0))

            return results
        except Exception:
            return []

