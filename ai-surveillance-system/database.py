from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class AlertEvent:
    type: str
    image: str
    name: str
    time_iso: str


class AlertDatabase:
    """
    SQLite-backed persistence for alert events.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT NOT NULL,
                    type TEXT NOT NULL,
                    image TEXT NOT NULL,
                    name TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(time)")
            conn.commit()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    def insert_alert(self, event_type: str, image_path: str | Path, name: str) -> int:
        image_path = str(image_path)
        now_iso = self._now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO alerts (time, type, image, name)
                VALUES (?, ?, ?, ?)
                """,
                (now_iso, event_type, image_path, name),
            )
            conn.commit()
            return int(cur.lastrowid)

    def fetch_last_alert(self) -> Optional[AlertEvent]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT time, type, image, name
                FROM alerts
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
            if not row:
                return None
            time_iso, event_type, image, name = row
            return AlertEvent(type=event_type, image=image, name=name, time_iso=time_iso)

    def count_alerts(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()
            return int(row[0]) if row else 0

    def fetch_alerts(self, limit: int = 100, offset: int = 0) -> list[AlertEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT time, type, image, name
                FROM alerts
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            return [
                AlertEvent(type=r[1], image=r[2], name=r[3], time_iso=r[0])
                for r in rows
            ]

    def fetch_all_alerts(self) -> Iterable[AlertEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT time, type, image, name
                FROM alerts
                ORDER BY id DESC
                """
            ).fetchall()
            return [
                AlertEvent(type=r[1], image=r[2], name=r[3], time_iso=r[0])
                for r in rows
            ]

