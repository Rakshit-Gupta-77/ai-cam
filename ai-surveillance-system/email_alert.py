from __future__ import annotations

import mimetypes
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SMTPConfig:
    host: str
    port: int
    user: str
    password: str
    sender: str
    recipients: list[str]
    use_tls: bool = True


class EmailAlerter:
    def __init__(self, config: SMTPConfig):
        self.config = config

    @staticmethod
    def from_env(
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        sender: Optional[str] = None,
        recipients: Optional[list[str]] = None,
    ) -> SMTPConfig:
        host = host or os.environ.get("SMTP_HOST", "").strip()
        port = port or int(os.environ.get("SMTP_PORT", "587").strip())
        user = user or os.environ.get("SMTP_USER", "").strip()
        password = password or os.environ.get("SMTP_PASS", "").strip()
        sender = sender or os.environ.get("SMTP_FROM", "").strip()
        recipients_env = os.environ.get("SMTP_TO", "").strip()
        recipients = recipients or [r.strip() for r in recipients_env.split(",") if r.strip()]

        if not host or not user or not password or not sender or not recipients:
            raise ValueError(
                "SMTP is not configured. Please set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_TO."
            )

        return SMTPConfig(
            host=host,
            port=port,
            user=user,
            password=password,
            sender=sender,
            recipients=recipients,
            use_tls=True,
        )

    def send_alert(self, *, subject: str, body: str, image_path: str | Path) -> None:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Email alert image not found: {image_path}")

        msg = EmailMessage()
        msg["From"] = self.config.sender
        msg["To"] = ", ".join(self.config.recipients)
        msg["Subject"] = subject
        msg.set_content(body)

        ctype, encoding = mimetypes.guess_type(str(image_path))
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"

        maintype, subtype = ctype.split("/", 1)
        with image_path.open("rb") as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=image_path.name)

        with smtplib.SMTP(self.config.host, self.config.port, timeout=30) as server:
            if self.config.use_tls:
                server.starttls()
            server.login(self.config.user, self.config.password)
            server.send_message(msg)

