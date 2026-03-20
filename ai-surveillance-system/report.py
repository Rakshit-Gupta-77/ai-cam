from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from database import AlertDatabase


@dataclass(frozen=True)
class ReportOptions:
    title: str = "AI Surveillance Alerts Report"


class PDFReportGenerator:
    def __init__(self, db: AlertDatabase, report_options: Optional[ReportOptions] = None):
        self.db = db
        self.options = report_options or ReportOptions()

    @staticmethod
    def _truncate(s: str, max_len: int) -> str:
        if len(s) <= max_len:
            return s
        return s[: max_len - 3] + "..."

    def generate(self, output_pdf_path: str | Path, limit: int = 200) -> Path:
        output_pdf_path = Path(output_pdf_path)
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

        alerts = self.db.fetch_alerts(limit=limit, offset=0)

        c = canvas.Canvas(str(output_pdf_path), pagesize=letter)
        width, height = letter
        margin = 48
        y = height - margin

        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y, self.options.title)
        y -= 24

        c.setFont("Helvetica", 10)
        c.drawString(margin, y, f"Total alerts (requested): {len(alerts)}")
        y -= 16

        # Simple text report to keep dependencies minimal.
        for a in alerts:
            if y < margin + 60:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - margin

            line = f"{a.time_iso} | {a.type} | {a.name}"
            c.drawString(margin, y, self._truncate(line, max_len=100))
            y -= 14

        c.save()
        return output_pdf_path

