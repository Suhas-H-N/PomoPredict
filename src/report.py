"""
Generates a one-page PDF report for a single prediction: the photo, the
predicted disease + confidence, estimated severity, and suggested treatment.
"""

import json
import os

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reports")


def build_pdf_report(prediction_row, disease_info):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f"report_{prediction_row['id']}.pdf")

    c = canvas.Canvas(out_path, pagesize=LETTER)
    width, height = LETTER
    margin = 0.75 * inch
    y = height - margin

    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, "PomoPredict Diagnosis Report")
    y -= 0.35 * inch

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated: {prediction_row['created_at']}")
    y -= 0.4 * inch

    img_path = prediction_row.get("image_path")
    if img_path and os.path.exists(img_path):
        try:
            img_w = 3.0 * inch
            img_h = 3.0 * inch
            c.drawImage(img_path, margin, y - img_h, width=img_w, height=img_h,
                        preserveAspectRatio=True, anchor='n')
            y -= img_h + 0.3 * inch
        except Exception:
            pass  # if the image can't be embedded, still produce the rest of the report

    c.setFont("Helvetica-Bold", 14)
    label = prediction_row["top_label"].replace("_", " ")
    c.drawString(margin, y, f"Diagnosis: {label}")
    y -= 0.25 * inch

    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Confidence: {prediction_row['top_confidence'] * 100:.1f}%")
    y -= 0.22 * inch

    if prediction_row.get("severity"):
        c.drawString(margin, y, f"Estimated severity: {prediction_row['severity']}")
        y -= 0.22 * inch

    if prediction_row.get("model_version"):
        c.drawString(margin, y, f"Model version: {prediction_row['model_version']}")
        y -= 0.3 * inch

    info = disease_info.get(prediction_row["top_label"], {})
    if info.get("note"):
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, "Notes:")
        y -= 0.2 * inch
        c.setFont("Helvetica", 10)
        y = _draw_wrapped(c, info["note"], margin, y, width - 2 * margin)
        y -= 0.15 * inch

    if info.get("treatment"):
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, "Suggested treatment:")
        y -= 0.2 * inch
        c.setFont("Helvetica", 10)
        y = _draw_wrapped(c, info["treatment"], margin, y, width - 2 * margin)
        y -= 0.15 * inch

    all_preds = json.loads(prediction_row["all_predictions_json"])
    if all_preds:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, "All predictions:")
        y -= 0.2 * inch
        c.setFont("Helvetica", 10)
        for name, conf in all_preds:
            c.drawString(margin + 0.1 * inch, y, f"{name.replace('_', ' ')}: {conf*100:.1f}%")
            y -= 0.18 * inch

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(margin, margin / 2,
                 "Educational demo — not a substitute for advice from a certified agronomist.")

    c.save()
    return out_path


def _draw_wrapped(c, text, x, y, max_width, font="Helvetica", size=10, leading=13):
    words = text.split()
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if c.stringWidth(trial, font, size) > max_width:
            c.drawString(x, y, line)
            y -= leading
            line = word
        else:
            line = trial
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y
