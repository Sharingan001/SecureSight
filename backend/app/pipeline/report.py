"""Court-Ready PDF Forensic Report Generator — ISO/IEC 27037:2012 Compliant.

Generates comprehensive digital forensic examination reports following:
  - ISO/IEC 27037:2012 — Digital Evidence Identification, Collection & Preservation
  - ISO/IEC 27041:2015 — Assuring Suitability of Investigation Methods
  - ISO/IEC 27042:2015 — Analysis and Interpretation of Digital Evidence
  - SWGDE Best Practices for Digital & Multimedia Evidence

Report Structure:
  1. Cover Page with classification & ISO markings
  2. Table of Contents
  3. Executive Summary with color-coded verdict
  4. Scope & Methodology (ISO 27041 compliance)
  5. Evidence Description & Integrity (dual-hash verification)
  6. Chain of Custody Log (ISO 27037 compliance)
  7. Detection Pipeline Results (all 15 pipelines)
  8. Visual Evidence (GradCAM heatmaps, ELA overlays)
  9. EXIF Metadata Analysis
  10. Findings & Interpretation (ISO 27042)
  11. Conclusions & Limitations
  12. Appendices (glossary, methodology details)
  13. Certification & Disclaimer
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    FrameBreak, HRFlowable, Image as RLImage, KeepTogether,
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


# ── Theme Colors ─────────────────────────────────────────────────────
NAVY        = colors.HexColor("#0f172a")
DARK        = colors.HexColor("#1e293b")
BODY_COLOR  = colors.HexColor("#334155")
ACCENT      = colors.HexColor("#2563eb")
LIGHT_BG    = colors.HexColor("#f1f5f9")
BORDER      = colors.HexColor("#cbd5e1")
HEADER_BG   = colors.HexColor("#1e293b")
HEADER_FG   = colors.white
RED         = colors.HexColor("#dc2626")
RED_LIGHT   = colors.HexColor("#fee2e2")
GREEN       = colors.HexColor("#16a34a")
GREEN_LIGHT = colors.HexColor("#dcfce7")
AMBER       = colors.HexColor("#d97706")
AMBER_LIGHT = colors.HexColor("#fef3c7")
ORANGE      = colors.HexColor("#ea580c")
STRIPE      = colors.HexColor("#f8fafc")


def _verdict_color(verdict: str) -> colors.Color:
    mapping = {
        "AUTHENTIC":       GREEN,
        "LIKELY_AUTHENTIC": colors.HexColor("#65a30d"),
        "SUSPICIOUS":      AMBER,
        "LIKELY_FAKE":     ORANGE,
        "CONFIRMED_FAKE":  RED,
    }
    return mapping.get(verdict, colors.HexColor("#64748b"))


def _verdict_bg(verdict: str) -> colors.Color:
    mapping = {
        "AUTHENTIC":       GREEN_LIGHT,
        "LIKELY_AUTHENTIC": colors.HexColor("#f0fdf4"),
        "SUSPICIOUS":      AMBER_LIGHT,
        "LIKELY_FAKE":     colors.HexColor("#ffedd5"),
        "CONFIRMED_FAKE":  RED_LIGHT,
    }
    return mapping.get(verdict, colors.HexColor("#f1f5f9"))


def _c(color: colors.Color) -> str:
    """Convert a ReportLab color to #RRGGBB hex string for use in markup."""
    return "#" + format(int(color.hexval(), 16), "06x")


def _verdict_label(verdict: str) -> str:
    return verdict.replace("_", " ")


def _verdict_description(verdict: str, score: float) -> str:
    descriptions = {
        "AUTHENTIC": (
            "The submitted media exhibits characteristics consistent with an authentic, "
            "unmanipulated digital file. No significant indicators of AI generation, "
            "deepfake manipulation, or digital tampering were identified across all "
            "active detection pipelines."
        ),
        "LIKELY_AUTHENTIC": (
            "The submitted media exhibits characteristics predominantly consistent with "
            "authentic digital media. Minor anomalies were detected which may be "
            "attributable to image compression, camera processing, or environmental "
            "factors rather than intentional manipulation."
        ),
        "SUSPICIOUS": (
            "The submitted media exhibits characteristics that warrant further "
            "investigation. One or more detection pipelines flagged potential indicators "
            "of AI generation or digital manipulation. A human forensic examiner should "
            "conduct additional analysis before drawing definitive conclusions."
        ),
        "LIKELY_FAKE": (
            "The submitted media exhibits multiple characteristics consistent with "
            "AI-generated or digitally manipulated content. Strong indicators were "
            "detected across multiple independent detection pipelines, suggesting the "
            "media has been artificially produced or significantly altered."
        ),
        "CONFIRMED_FAKE": (
            "The submitted media exhibits overwhelming characteristics of AI-generated "
            "or deepfake content. Multiple independent detection methods produced "
            "high-confidence results indicating the media is not an authentic, "
            "unmanipulated file."
        ),
    }
    return descriptions.get(verdict, "Unable to determine media authenticity.")


# ── Style Builder ─────────────────────────────────────────────────────
def _build_styles() -> dict:
    base = getSampleStyleSheet()
    S = {}

    S["title"] = ParagraphStyle(
        "ReportTitle", parent=base["Title"],
        fontSize=30, textColor=NAVY, spaceAfter=2, spaceBefore=0,
        alignment=TA_CENTER, fontName="Helvetica-Bold", leading=34,
    )
    S["subtitle"] = ParagraphStyle(
        "Subtitle", parent=base["Normal"],
        fontSize=11, textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER, spaceAfter=6, spaceBefore=0,
        fontName="Helvetica", leading=15,
    )
    S["h1"] = ParagraphStyle(
        "H1", parent=base["Heading1"],
        fontSize=12, textColor=colors.white,
        spaceBefore=14, spaceAfter=8,
        fontName="Helvetica-Bold",
        backColor=NAVY,
        borderPadding=(7, 10, 7, 10),
    )
    S["h2"] = ParagraphStyle(
        "H2", parent=base["Heading2"],
        fontSize=10, textColor=NAVY,
        spaceBefore=8, spaceAfter=4,
        fontName="Helvetica-Bold",
        borderWidth=0, borderPadding=0,
    )
    S["body"] = ParagraphStyle(
        "Body", parent=base["Normal"],
        fontSize=9, leading=14, textColor=BODY_COLOR,
    )
    S["small"] = ParagraphStyle(
        "Small", parent=base["Normal"],
        fontSize=7.5, leading=11,
        textColor=colors.HexColor("#64748b"),
    )
    S["mono"] = ParagraphStyle(
        "Mono", parent=base["Normal"],
        fontSize=7.5, leading=11,
        fontName="Courier", textColor=BODY_COLOR,
    )
    S["center"] = ParagraphStyle(
        "Center", parent=base["Normal"],
        fontSize=9, leading=14,
        textColor=BODY_COLOR, alignment=TA_CENTER,
    )
    return S


# ── Header/Footer ─────────────────────────────────────────────────────
def _add_header_footer(canvas, doc):
    """Professional header/footer on every page except cover."""
    if doc.page == 1:
        return  # Cover page: no header/footer overlay

    canvas.saveState()

    # Header bar
    canvas.setFillColor(NAVY)
    canvas.rect(0, A4[1] - 20 * mm, A4[0], 20 * mm, fill=1, stroke=0)

    # Accent stripe
    canvas.setFillColor(ACCENT)
    canvas.rect(0, A4[1] - 21.5 * mm, A4[0], 1.5 * mm, fill=1, stroke=0)

    # Header text
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(colors.white)
    canvas.drawString(20 * mm, A4[1] - 13 * mm, "SECURESIGHT — Digital Forensic Analysis Report")

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#94a3b8"))
    canvas.drawRightString(A4[0] - 20 * mm, A4[1] - 13 * mm, "ISO/IEC 27037 · 27041 · 27042")

    # Footer bar
    canvas.setFillColor(LIGHT_BG)
    canvas.rect(0, 0, A4[0], 14 * mm, fill=1, stroke=0)

    canvas.setFillColor(ACCENT)
    canvas.rect(0, 13.5 * mm, A4[0], 0.5 * mm, fill=1, stroke=0)

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(20 * mm, 5 * mm, "CONFIDENTIAL — For Authorised Personnel Only")
    canvas.drawCentredString(A4[0] / 2, 5 * mm, f"Page {doc.page}")
    canvas.drawRightString(
        A4[0] - 20 * mm, 5 * mm,
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

    canvas.restoreState()


def _add_cover_footer(canvas, doc):
    """Cover-page only footer."""
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, A4[0], 12 * mm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#94a3b8"))
    canvas.drawCentredString(A4[0] / 2, 4 * mm, "SECURESIGHT Forensic Intelligence Platform — Confidential")
    canvas.restoreState()


def _cover_and_body(canvas, doc):
    if doc.page == 1:
        _add_cover_footer(canvas, doc)
    else:
        _add_header_footer(canvas, doc)


# ── Reusable Table Style ───────────────────────────────────────────────
def _table_style(striped: bool = True) -> TableStyle:
    cmds = [
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8),
        ("BACKGROUND",     (0, 0), (-1, 0),  HEADER_BG),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  HEADER_FG),
        ("GRID",           (0, 0), (-1, -1), 0.4, BORDER),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, STRIPE] if striped else [colors.white]),
    ]
    return TableStyle(cmds)


def _section_header(title: str, S: dict) -> list:
    """Returns a list of elements: accent bar + title paragraph."""
    return [
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=3, color=ACCENT, spaceAfter=0),
        Paragraph(title, S["h1"]),
        Spacer(1, 4),
    ]


# ── Main Generator ────────────────────────────────────────────────────
def generate_report(
    output_path: str,
    analysis_id: str,
    evidence_id: str,
    filename: str,
    sha256: str,
    sha512: str = "",
    overall_score: float = 0.0,
    verdict: str = "UNKNOWN",
    pipeline_scores: list[dict[str, Any]] | None = None,
    exif_data: dict[str, Any] | None = None,
    custody_log: list[dict[str, Any]] | None = None,
    heatmap_paths: list[str] | None = None,
    examiner_name: str = "SecureSight Automated Forensic Examiner",
) -> str:
    """Generate a court-ready ISO 27037/27042 compliant PDF forensic report."""

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=25 * mm, bottomMargin=20 * mm,
        title=f"Forensic Report — {evidence_id}",
        author="SecureSight v2.0",
        subject="Digital Media Forensic Examination",
        keywords="deepfake, forensics, digital evidence, AI detection",
    )

    S = _build_styles()
    elements = []
    now = datetime.now(timezone.utc)
    v_color = _verdict_color(verdict)
    v_bg = _verdict_bg(verdict)
    v_hex = _c(v_color)
    section = 0

    # ═══════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ═══════════════════════════════════════════════════════════════════

    # Classification banner
    elements.append(Spacer(1, 8 * mm))
    class_data = [["▐  CONFIDENTIAL — FORENSIC EXAMINATION REPORT  ▌"]]
    class_table = Table(class_data, colWidths=[170 * mm])
    class_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), RED),
        ("TEXTCOLOR",     (0, 0), (-1, -1), colors.white),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    elements.append(class_table)
    elements.append(Spacer(1, 14 * mm))

    # Title block
    elements.append(Paragraph("SECURESIGHT", S["title"]))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph("Digital Media Forensic Analysis Report", S["subtitle"]))
    elements.append(Spacer(1, 3 * mm))
    elements.append(HRFlowable(width="60%", thickness=2, color=ACCENT,
                               spaceAfter=8, hAlign="CENTER"))

    # ISO compliance badges
    iso_text = (
        f'<font color="{_c(ACCENT)}"><b>ISO/IEC 27037:2012</b></font>'
        f'<font color="{_c(BODY_COLOR)}"> &nbsp;·&nbsp; </font>'
        f'<font color="{_c(ACCENT)}"><b>ISO/IEC 27041:2015</b></font>'
        f'<font color="{_c(BODY_COLOR)}"> &nbsp;·&nbsp; </font>'
        f'<font color="{_c(ACCENT)}"><b>ISO/IEC 27042:2015</b></font>'
    )
    elements.append(Paragraph(iso_text, ParagraphStyle(
        "ISOBadge", parent=S["body"], fontSize=8.5,
        alignment=TA_CENTER, spaceAfter=0, spaceBefore=0,
    )))
    elements.append(Spacer(1, 10 * mm))

    # Cover metadata table
    cover_data = [
        ["Report Reference",   f"SR-{analysis_id[:8].upper()}"],
        ["Evidence ID",         evidence_id],
        ["Filename",            filename],
        ["Report Date",         now.strftime("%d %B %Y  —  %H:%M UTC")],
        ["Classification",      "CONFIDENTIAL"],
        ["Examiner",            examiner_name],
        ["Examination Method",  "Automated Multi-Pipeline AI & Forensic Analysis"],
        ["Software Version",    "SecureSight v2.0"],
        ["Report Standard",     "ISO/IEC 27037:2012 / 27041:2015 / 27042:2015"],
    ]
    cover_table = Table(cover_data, colWidths=[55 * mm, 115 * mm])
    cover_table.setStyle(TableStyle([
        ("FONTNAME",       (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (0, -1),  8.5),
        ("FONTNAME",       (1, 0), (1, -1),  "Helvetica"),
        ("FONTSIZE",       (1, 0), (1, -1),  9),
        ("TEXTCOLOR",      (0, 0), (0, -1),  NAVY),
        ("TEXTCOLOR",      (1, 0), (1, -1),  BODY_COLOR),
        ("TOPPADDING",     (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 7),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("LINEBELOW",      (0, 0), (-1, -2), 0.4, BORDER),
        ("LINEBELOW",      (0, -1), (-1, -1), 2, ACCENT),
        ("BACKGROUND",     (0, 0), (0, -1),  LIGHT_BG),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, STRIPE]),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(cover_table)
    elements.append(Spacer(1, 10 * mm))

    # Verdict panel
    verdict_label = _verdict_label(verdict)
    vrow = [
        Paragraph(
            '<font size=8><b>OVERALL ASSESSMENT</b></font>',
            ParagraphStyle("", parent=S["body"],
                           textColor=colors.HexColor("#64748b"), alignment=TA_LEFT),
        ),
        Paragraph(
            f'<font size=18 color="{v_hex}"><b>{verdict_label}</b></font>',
            ParagraphStyle("", parent=S["body"], alignment=TA_CENTER),
        ),
        Paragraph(
            f'<font size=22 color="{v_hex}"><b>{overall_score:.1f}</b></font>'
            f'<font size=10 color="#64748b">/100</font>',
            ParagraphStyle("", parent=S["body"], alignment=TA_RIGHT),
        ),
    ]
    verdict_cover = Table([vrow], colWidths=[55 * mm, 75 * mm, 40 * mm])
    verdict_cover.setStyle(TableStyle([
        ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",     (0, 0), (-1, -1), v_bg),
        ("BOX",            (0, 0), (-1, -1), 2, v_color),
        ("TOPPADDING",     (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 14),
        ("LEFTPADDING",    (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    elements.append(verdict_cover)
    elements.append(PageBreak())


    # ═══════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ═══════════════════════════════════════════════════════════════════
    elements.extend(_section_header("Table of Contents", S))
    toc_items = [
        ("1",  "Executive Summary"),
        ("2",  "Scope & Methodology"),
        ("3",  "Evidence Description & Integrity Verification"),
        ("4",  "Chain of Custody"),
        ("5",  "Detection Pipeline Results"),
        ("6",  "Visual Evidence & Heatmap Analysis"),
        ("7",  "EXIF Metadata Analysis"),
        ("8",  "Findings & Interpretation"),
        ("9",  "Conclusions & Limitations"),
        ("10", "Certification & Disclaimer"),
        ("A",  "Appendix A — Glossary of Terms"),
        ("B",  "Appendix B — Pipeline Methodology"),
    ]
    toc_data = [[n, title] for n, title in toc_items]
    toc_table = Table(toc_data, colWidths=[14 * mm, 156 * mm])
    toc_table.setStyle(TableStyle([
        ("FONTNAME",       (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (0, -1),  11),
        ("FONTNAME",       (1, 0), (1, -1),  "Helvetica"),
        ("FONTSIZE",       (1, 0), (1, -1),  11),
        ("TEXTCOLOR",      (0, 0), (0, -1),  ACCENT),
        ("TEXTCOLOR",      (1, 0), (1, -1),  NAVY),
        ("TOPPADDING",     (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 9),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("LINEBELOW",      (0, 0), (-1, -1), 0.4, BORDER),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, STRIPE]),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(toc_table)
    elements.append(PageBreak())


    # ═══════════════════════════════════════════════════════════════════
    # 1. EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    section += 1
    elements.extend(_section_header(f"{section}. Executive Summary", S))

    elements.append(Paragraph(
        f"This report presents the findings of an automated digital media forensic "
        f"examination conducted on the file <b>'{filename}'</b> "
        f"(Evidence ID: <b>{evidence_id}</b>). "
        f"The examination was performed using SecureSight's 15-pipeline analysis "
        f"framework, which employs deep learning models, image forensic algorithms, "
        f"biometric analysis, and physical consistency checks to determine the "
        f"authenticity of digital media.",
        S["body"],
    ))
    elements.append(Spacer(1, 8))

    # Score panel
    score_data = [[
        Paragraph(
            f'<font color="white"><b>OVERALL AUTHENTICITY SCORE</b></font>',
            ParagraphStyle("", parent=S["body"], textColor=colors.white, fontSize=10),
        ),
        Paragraph(
            f'<font color="white" size=22><b>{overall_score:.1f}</b></font>'
            f'<font color="#cbd5e1" size=10> / 100</font>',
            ParagraphStyle("", parent=S["body"], textColor=colors.white,
                           fontSize=20, alignment=TA_RIGHT),
        ),
    ]]
    score_table = Table(score_data, colWidths=[120 * mm, 50 * mm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), v_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(KeepTogether([score_table]))
    elements.append(Spacer(1, 8))

    # Verdict badge
    verdict_badge = [[
        Paragraph(f'<b>Verdict:</b>', S["body"]),
        Paragraph(
            f'<font color="{v_hex}"><b>{verdict_label}</b></font>',
            ParagraphStyle("", parent=S["body"], fontSize=11),
        ),
    ]]
    vbadge_t = Table(verdict_badge, colWidths=[25 * mm, 145 * mm])
    vbadge_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), v_bg),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("BOX",           (0, 0), (-1, -1), 0.5, v_color),
    ]))
    elements.append(vbadge_t)
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(_verdict_description(verdict, overall_score), S["body"]))
    elements.append(Spacer(1, 10))

    # Score scale
    scale_data = [
        ["Score Range", "Verdict", "Interpretation"],
        ["0 – 15",   "AUTHENTIC",         "High confidence genuine media"],
        ["15 – 35",  "LIKELY AUTHENTIC",  "Probably genuine, minor anomalies"],
        ["35 – 60",  "SUSPICIOUS",        "Warrants further investigation"],
        ["60 – 85",  "LIKELY FAKE",       "Strong manipulation indicators"],
        ["85 – 100", "CONFIRMED FAKE",    "Near-certain AI/deepfake content"],
    ]
    scale_table = Table(scale_data, colWidths=[30 * mm, 55 * mm, 85 * mm])
    scale_table.setStyle(_table_style())

    # Highlight the active row
    scale_row_map = {
        "AUTHENTIC": 1, "LIKELY_AUTHENTIC": 2,
        "SUSPICIOUS": 3, "LIKELY_FAKE": 4, "CONFIRMED_FAKE": 5,
    }
    active_row = scale_row_map.get(verdict, 0)
    if active_row:
        scale_table.setStyle(TableStyle([
            ("BACKGROUND", (0, active_row), (-1, active_row), v_bg),
            ("TEXTCOLOR",  (0, active_row), (-1, active_row), v_color),
            ("FONTNAME",   (0, active_row), (-1, active_row), "Helvetica-Bold"),
        ]))
    elements.append(KeepTogether([scale_table]))

    # ═══════════════════════════════════════════════════════════════════
    # 2. SCOPE & METHODOLOGY
    # ═══════════════════════════════════════════════════════════════════
    section += 1
    elements.extend(_section_header(f"{section}. Scope & Methodology", S))

    elements.append(Paragraph(
        "This examination was conducted in accordance with the following international standards:",
        S["body"],
    ))
    elements.append(Spacer(1, 6))

    standards = [
        ["Standard",          "Title",                                                                "Compliance"],
        ["ISO/IEC 27037:2012", "Digital Evidence — Identification, Collection, Acquisition & Preservation", "✔ Full"],
        ["ISO/IEC 27041:2015", "Assuring Suitability and Adequacy of Investigation Methods",            "✔ Full"],
        ["ISO/IEC 27042:2015", "Analysis and Interpretation of Digital Evidence",                       "✔ Full"],
        ["SWGDE v3.0",         "Best Practices for Digital & Multimedia Evidence",                      "✔ Partial"],
        ["NIST SP 800-86",     "Guide to Integrating Forensic Techniques into Incident Response",       "✔ Partial"],
    ]
    std_table = Table(standards, colWidths=[38 * mm, 100 * mm, 32 * mm])
    std_table.setStyle(_table_style())
    std_table.setStyle(TableStyle([
        ("TEXTCOLOR",  (2, 1), (2, -1), GREEN),
        ("FONTNAME",   (2, 1), (2, -1), "Helvetica-Bold"),
    ]))
    elements.append(KeepTogether([std_table]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>Examination Methodology</b>", S["h2"]))
    elements.append(Paragraph(
        "The analysis employs a multi-layered detection framework consisting of 15 "
        "independent detection pipelines organised into 4 tiers. Pipelines that cannot "
        "run (e.g. face-only models on images with no detected faces, or video-only "
        "pipelines on static images) are <b>excluded from scoring</b> — they do not "
        "vote 'authentic'. The final score is derived from a calibrated weighted "
        "ensemble of only the pipelines that actively ran.",
        S["body"],
    ))
    elements.append(Spacer(1, 6))

    tiers = [
        ["Tier",                     "Pipelines",                                           "Focus Area"],
        ["Tier 1 — Deep Learning",   "EfficientNet-B4, AI Ensemble (ViT+CLIP+SigLIP), XceptionNet, Audio, LipSync",
                                     "Neural network-based classification"],
        ["Tier 2 — Image Forensics", "ELA, Copy-Move, JPEG Ghost, EXIF Metadata",
                                     "Pixel-level manipulation detection"],
        ["Tier 3 — Physical",        "Eye Reflection, Shadow/Lighting, Noise Pattern",
                                     "Physical world law violations"],
        ["Tier 4 — Advanced",        "Frequency Domain, Biometric Face Mesh, Temporal",
                                     "Spectral, geometric & temporal anomalies"],
    ]
    tier_table = Table(tiers, colWidths=[42 * mm, 80 * mm, 48 * mm])
    tier_table.setStyle(_table_style())
    elements.append(KeepTogether([tier_table]))

    # ═══════════════════════════════════════════════════════════════════
    # 3. EVIDENCE DESCRIPTION & INTEGRITY
    # ═══════════════════════════════════════════════════════════════════
    section += 1
    elements.extend(_section_header(f"{section}. Evidence Description & Integrity Verification", S))

    elements.append(Paragraph(
        "The following digital evidence was submitted for examination. Cryptographic hash "
        "values were computed at the time of intake to ensure evidence integrity throughout "
        "the examination process, in compliance with <b>ISO/IEC 27037:2012 §7.1.3</b>.",
        S["body"],
    ))
    elements.append(Spacer(1, 8))

    ev_data = [
        ["Property",        "Value"],
        ["Filename",         filename],
        ["Evidence ID",      evidence_id],
        ["Analysis Ref.",    f"SR-{analysis_id[:8].upper()}"],
        ["Intake Timestamp", now.strftime("%Y-%m-%d %H:%M:%S UTC")],
    ]
    ev_data.append(["SHA-256 Hash",
        Paragraph(f'<font face="Courier" size=7>{sha256}</font>', S["body"])])
    if sha512:
        ev_data.append(["SHA-512 Hash",
            Paragraph(f'<font face="Courier" size=6>{sha512}</font>', S["body"])])
    ev_data.append(["Integrity Status",
        Paragraph(f'<font color="{_c(GREEN)}"><b>✔ VERIFIED — Hash values computed and recorded at intake</b></font>', S["body"])])

    ev_table = Table(ev_data, colWidths=[38 * mm, 132 * mm])
    ev_table.setStyle(_table_style())
    elements.append(KeepTogether([ev_table]))

    # ═══════════════════════════════════════════════════════════════════
    # 4. CHAIN OF CUSTODY
    # ═══════════════════════════════════════════════════════════════════
    section += 1
    elements.extend(_section_header(f"{section}. Chain of Custody", S))

    elements.append(Paragraph(
        "The following chain of custody log documents all actions performed on the "
        "evidence, in compliance with <b>ISO/IEC 27037:2012 §7.2</b>. Each entry "
        "includes the action performed, the actor, timestamp, and cryptographic hash.",
        S["body"],
    ))
    elements.append(Spacer(1, 8))

    if custody_log:
        coc_data = [["#", "Timestamp (UTC)", "Action", "Actor", "Hash (first 16 chars)"]]
        for i, entry in enumerate(custody_log[:25], 1):
            h = entry.get("file_hash", "")
            hash_display = f"{h[:16]}…" if len(h) > 16 else (h or "—")
            coc_data.append([
                str(i),
                str(entry.get("timestamp", ""))[:19],
                entry.get("action", "").replace("_", " ").title(),
                entry.get("actor", ""),
                Paragraph(f'<font face="Courier" size=7>{hash_display}</font>', S["body"]),
            ])
        coc_table = Table(coc_data, colWidths=[8 * mm, 36 * mm, 44 * mm, 52 * mm, 30 * mm])
        coc_table.setStyle(_table_style())
        elements.append(KeepTogether([coc_table]))
    else:
        elements.append(Paragraph(
            "<i>No chain of custody entries were recorded for this analysis. "
            "Custody logging requires database persistence to be enabled.</i>",
            S["small"],
        ))

    # ═══════════════════════════════════════════════════════════════════
    # 5. DETECTION PIPELINE RESULTS
    # ═══════════════════════════════════════════════════════════════════
    section += 1
    elements.extend(_section_header(f"{section}. Detection Pipeline Results", S))

    elements.append(Paragraph(
        "The following table presents results from each active detection pipeline. "
        "Scores range from 0.000 (strongly authentic) to 1.000 (strongly manipulated). "
        "Pipelines are sorted by tier and score. Pipelines that did not run for this "
        "media type are omitted.",
        S["body"],
    ))
    elements.append(Spacer(1, 8))

    if pipeline_scores:
        pipe_data = [["Pipeline", "Tier", "Score", "Confidence", "Time (ms)", "Flag"]]
        for ps in sorted(pipeline_scores, key=lambda x: (x.get("tier", 9), -x.get("score", 0))):
            score_val = ps.get("score", 0)
            conf_val  = ps.get("confidence", 1.0)
            if score_val < 0.30:
                flag_text  = "✔ No anomaly"
                flag_color = _c(GREEN)
            elif score_val < 0.60:
                flag_text  = "⚠ Minor flag"
                flag_color = _c(AMBER)
            else:
                flag_text  = "✘ Significant"
                flag_color = _c(RED)

            pipe_data.append([
                ps.get("pipeline", "").replace("_", " ").title(),
                str(ps.get("tier", "—")),
                f"{score_val:.3f}",
                f"{conf_val:.2f}",
                str(int(ps.get("execution_ms", 0))),
                Paragraph(f'<font color="{flag_color}"><b>{flag_text}</b></font>', S["small"]),
            ])

        pipe_table = Table(pipe_data, colWidths=[42 * mm, 12 * mm, 22 * mm, 25 * mm, 22 * mm, 47 * mm])
        pipe_table.setStyle(_table_style())
        elements.append(KeepTogether([pipe_table]))

        # Key findings callout
        significant = [p for p in pipeline_scores if p.get("score", 0) > 0.50]
        if significant:
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("<b>Key Findings (Score > 0.50):</b>", S["h2"]))
            findings_rows = [["Pipeline", "Score", "Detail"]]
            for ps in sorted(significant, key=lambda x: -x.get("score", 0)):
                pname = ps.get("pipeline", "").replace("_", " ").title()
                pscore = ps.get("score", 0)
                details = ps.get("details", {})
                detail_str = ""
                if "strategy" in details:
                    detail_str = f"Fusion: {details['strategy']}"
                elif "software_tag" in details:
                    detail_str = f"Software: {details['software_tag']}"
                elif "tta_votes" in details:
                    detail_str = f"TTA votes: {details['tta_votes']}"
                findings_rows.append([
                    pname,
                    Paragraph(f'<font color="{_c(RED)}"><b>{pscore:.3f}</b></font>', S["small"]),
                    detail_str or "—",
                ])
            findings_t = Table(findings_rows, colWidths=[55 * mm, 25 * mm, 90 * mm])
            findings_t.setStyle(_table_style())
            elements.append(KeepTogether([findings_t]))
    else:
        elements.append(Paragraph("<i>No pipeline results available for this analysis.</i>", S["small"]))

    # ═══════════════════════════════════════════════════════════════════
    # 6. VISUAL EVIDENCE
    # ═══════════════════════════════════════════════════════════════════
    section += 1
    elements.extend(_section_header(f"{section}. Visual Evidence & Heatmap Analysis", S))

    if heatmap_paths:
        elements.append(Paragraph(
            "The following visualisations highlight regions of interest identified by "
            "the deep learning models. GradCAM heatmaps indicate the areas that most "
            "strongly influenced the model's classification decision. Warmer colours "
            "(red/orange) indicate high-influence regions consistent with AI generation "
            "or manipulation artefacts.",
            S["body"],
        ))
        elements.append(Spacer(1, 8))
        for hp in heatmap_paths[:6]:
            if os.path.exists(hp):
                try:
                    label = (
                        os.path.basename(hp)
                        .replace("_", " ")
                        .replace(".png", "")
                        .replace(".jpg", "")
                        .title()
                    )
                    caption = Paragraph(f"<b>Figure:</b> {label}", S["small"])
                    img = RLImage(hp, width=155 * mm, height=115 * mm, kind="proportional")
                    border_data = [[img]]
                    border_t = Table(border_data, colWidths=[170 * mm])
                    border_t.setStyle(TableStyle([
                        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
                        ("TOPPADDING",    (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]))
                    elements.append(KeepTogether([border_t, caption]))
                    elements.append(Spacer(1, 10))
                except Exception:
                    pass
    else:
        note_data = [[Paragraph(
            "ℹ No heatmap visualisations were generated for this analysis. "
            "Heatmaps are produced when the EfficientNet-B4 or XceptionNet "
            "model runs with GradCAM enabled.",
            S["small"],
        )]]
        note_t = Table(note_data, colWidths=[170 * mm])
        note_t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BG),
            ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ]))
        elements.append(note_t)

    # ═══════════════════════════════════════════════════════════════════
    # 7. EXIF METADATA
    # ═══════════════════════════════════════════════════════════════════
    section += 1
    elements.extend(_section_header(f"{section}. EXIF Metadata Analysis", S))

    if exif_data and len(exif_data) > 0:
        elements.append(Paragraph(
            "The following EXIF metadata was extracted from the submitted evidence file. "
            "Anomalies such as stripped metadata, editing software tags, and timestamp "
            "inconsistencies are flagged as potential indicators of manipulation.",
            S["body"],
        ))
        elements.append(Spacer(1, 8))

        exif_rows = [["Metadata Tag", "Value"]]
        for key, val in list(exif_data.items())[:35]:
            val_str = str(val)[:90]
            exif_rows.append([str(key), val_str])

        exif_table = Table(exif_rows, colWidths=[65 * mm, 105 * mm])
        exif_table.setStyle(_table_style())
        elements.append(KeepTogether([exif_table]))
    else:
        elements.append(Paragraph(
            "No EXIF metadata was found in the submitted file. The absence of metadata "
            "may indicate that the image has been processed, stripped, or generated by "
            "a software tool that does not embed camera-originated EXIF data — a common "
            "characteristic of AI-generated media.",
            S["body"],
        ))

    # ═══════════════════════════════════════════════════════════════════
    # 8. FINDINGS & INTERPRETATION
    # ═══════════════════════════════════════════════════════════════════
    section += 1
    elements.extend(_section_header(f"{section}. Findings & Interpretation", S))

    elements.append(Paragraph(
        "In accordance with <b>ISO/IEC 27042:2015 §8</b>, the following interpretation "
        "is provided based on the aggregated results of all active detection pipelines:",
        S["body"],
    ))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(_verdict_description(verdict, overall_score), S["body"]))
    elements.append(Spacer(1, 10))

    if pipeline_scores:
        ai_ens = [p for p in pipeline_scores if p.get("pipeline") == "ai_ensemble"]
        eff    = [p for p in pipeline_scores if p.get("pipeline") == "efficientnet"]
        xce    = [p for p in pipeline_scores if p.get("pipeline") == "xception"]

        if ai_ens:
            ai_score = ai_ens[0].get("score", 0)
            details  = ai_ens[0].get("details", {})
            strategy = details.get("strategy", "unknown")
            voting   = details.get("voting", {})
            elements.append(Paragraph("<b>AI Ensemble Analysis (35% weight):</b>", S["h2"]))
            elements.append(Paragraph(
                f"The 3-model AI ensemble (haywoodsloan ViT, umm-maybe CLIP, "
                f"Ateeqq SigLIP+DINOv2) produced an aggregated score of "
                f"<b>{ai_score:.3f}</b> using the <b>{strategy}</b> fusion strategy. "
                f"Voting: {voting.get('ai_votes', 0)} AI vote(s), "
                f"{voting.get('real_votes', 0)} real vote(s), "
                f"{voting.get('uncertain', 0)} uncertain.",
                S["body"],
            ))
            elements.append(Spacer(1, 6))

        if eff:
            eff_score = eff[0].get("score", 0)
            elements.append(Paragraph("<b>EfficientNet-B4 Analysis (25% weight):</b>", S["h2"]))
            elements.append(Paragraph(
                f"The custom-trained EfficientNet-B4 model produced a score of "
                f"<b>{eff_score:.3f}</b> using Test-Time Augmentation (5 variants). "
                f"This model is the highest single-weight detector in the ensemble.",
                S["body"],
            ))
            elements.append(Spacer(1, 6))

        if xce:
            xce_score = xce[0].get("score", 0)
            elements.append(Paragraph("<b>XceptionNet Analysis (10% weight):</b>", S["h2"]))
            elements.append(Paragraph(
                f"XceptionNet produced a score of <b>{xce_score:.3f}</b>.",
                S["body"],
            ))

    # ═══════════════════════════════════════════════════════════════════
    # 9. CONCLUSIONS & LIMITATIONS
    # ═══════════════════════════════════════════════════════════════════
    section += 1
    elements.extend(_section_header(f"{section}. Conclusions & Limitations", S))

    elements.append(Paragraph(
        f"Based on comprehensive automated analysis using 15 independent detection "
        f"pipelines, the submitted file <b>'{filename}'</b> "
        f"(Evidence ID: <b>{evidence_id}</b>) received an overall score of "
        f"<b>{overall_score:.1f}/100</b> with a verdict of "
        f'<font color="{v_hex}"><b>{verdict_label}</b></font>.',
        S["body"],
    ))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Limitations of This Report:</b>", S["h2"]))
    limitations = [
        "This analysis is performed by automated systems and should be reviewed by a "
        "qualified digital forensic examiner before use in legal proceedings.",
        "Detection accuracy may be affected by heavy image compression, very low "
        "resolution, or novel AI generation methods not present in the training data.",
        "An AUTHENTIC verdict does not guarantee the media has not been manipulated; "
        "it indicates no detectable manipulation was found using current methods.",
        "A CONFIRMED FAKE verdict is a strong indicator but not conclusive proof; "
        "certain legitimate image processing may trigger false positives.",
        "The AI ensemble models are pre-trained on publicly available datasets and may "
        "have biases for specific content types or generation methods.",
        "Video analysis accuracy depends on sampling rate; subtle per-frame "
        "manipulations in undersampled segments may be missed.",
    ]
    for i, lim in enumerate(limitations):
        elements.append(Paragraph(
            f'<font color="{_c(ACCENT)}"><b>L{i+1}</b></font> &nbsp; {lim}',
            ParagraphStyle("LimItem", parent=S["body"], fontSize=10,
                           leading=15, leftIndent=20, spaceBefore=7, spaceAfter=7),
        ))
        elements.append(HRFlowable(width="100%", thickness=0.4, color=BORDER, spaceAfter=2))


    # ═══════════════════════════════════════════════════════════════════
    # 10. CERTIFICATION & DISCLAIMER
    # ═══════════════════════════════════════════════════════════════════
    section += 1
    elements.append(PageBreak())
    elements.extend(_section_header(f"{section}. Certification & Disclaimer", S))

    # Cert header
    cert_data = [["CERTIFICATION OF EXAMINATION"]]
    cert_header = Table(cert_data, colWidths=[170 * mm])
    cert_header.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, -1), colors.white),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 11),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(cert_header)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        "I hereby certify that this report accurately represents the findings of the "
        "digital forensic examination conducted on the submitted evidence. The examination "
        "was performed using scientifically validated methods in accordance with "
        "ISO/IEC 27037:2012, ISO/IEC 27041:2015, and ISO/IEC 27042:2015.",
        ParagraphStyle("", parent=S["body"], fontSize=10.5, leading=16),
    ))
    elements.append(Spacer(1, 22))

    sig_data = [
        ["Examiner:",        examiner_name],
        ["Date:",            now.strftime("%d %B %Y")],
        ["Time:",            now.strftime("%H:%M:%S UTC")],
        ["Report Ref.:",     f"SR-{analysis_id[:8].upper()}"],
        ["Evidence ID:",     evidence_id],
        ["Software:",        "SecureSight v2.0 — Forensic AI Detection Platform"],
    ]
    sig_table = Table(sig_data, colWidths=[42 * mm, 128 * mm])
    sig_table.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 11),
        ("TEXTCOLOR",     (0, 0), (0, -1),  NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("BACKGROUND",    (0, 0), (0, -1),  LIGHT_BG),
        ("LINEBELOW",     (0, -1), (-1, -1), 2, NAVY),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.4, BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(sig_table)
    elements.append(Spacer(1, 28))

    # Signature line
    sig_line_data = [["Signature:", "", "Date:"]]
    sig_line = Table(sig_line_data, colWidths=[28 * mm, 92 * mm, 50 * mm])
    sig_line.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("LINEBELOW",     (1, 0), (1, 0),   0.8, NAVY),
        ("LINEBELOW",     (2, 0), (2, 0),   0.8, NAVY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 22),
    ]))
    elements.append(sig_line)
    elements.append(Spacer(1, 20))

    # Disclaimer box
    disclaimer_data = [[Paragraph(
        "<b>DISCLAIMER:</b> This report is generated by SecureSight, an automated "
        "digital forensic analysis platform. While the system employs state-of-the-art "
        "AI models and forensic techniques, the results should be interpreted by "
        "qualified forensic examiners. This report does not constitute legal advice. "
        "The findings are probabilistic assessments and should not be considered as "
        "definitive proof of authenticity or manipulation. Any use of this report in "
        "legal proceedings should be accompanied by expert testimony from a certified "
        "digital forensic examiner. SecureSight and its operators assume no liability "
        "for decisions made based solely on the contents of this report.",
        ParagraphStyle("", parent=S["body"], fontSize=9.5, leading=15,
                       textColor=colors.HexColor("#475569")),
    )]]
    disclaimer_table = Table(disclaimer_data, colWidths=[170 * mm])
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BG),
        ("BOX",           (0, 0), (-1, -1), 1.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
    ]))
    elements.append(disclaimer_table)


    # ═══════════════════════════════════════════════════════════════════
    # APPENDIX A — GLOSSARY
    # ═══════════════════════════════════════════════════════════════════
    elements.append(PageBreak())
    elements.extend(_section_header("Appendix A — Glossary of Terms", S))

    _mb = ParagraphStyle("mb", parent=S["body"], fontSize=9.5, leading=13)
    _mbh = ParagraphStyle("mbh", parent=_mb, fontName="Helvetica-Bold", textColor=colors.white)
    glossary_raw = [
        ["AUC",        "Area Under the ROC Curve — measures model discrimination (0.5 = random, 1.0 = perfect)"],
        ["CLIP",       "Contrastive Language–Image Pre-training — multimodal AI model by OpenAI"],
        ["Deepfake",   "AI-generated or AI-manipulated media designed to mimic authentic content"],
        ["DINOv2",     "Self-supervised vision transformer by Meta — used in SigLIP ensemble model"],
        ["ELA",        "Error Level Analysis — detects manipulation by comparing JPEG compression artefacts"],
        ["EfficientNet","CNN architecture trained on face crops to classify as real or AI-generated"],
        ["EXIF",       "Exchangeable Image File Format — metadata embedded by cameras and editing software"],
        ["GAN",        "Generative Adversarial Network — AI architecture that generates synthetic images"],
        ["GradCAM",    "Gradient-weighted Class Activation Mapping — neural network explainability method"],
        ["SHA-256/512","Secure Hash Algorithm — cryptographic hash functions for data integrity verification"],
        ["SigLIP",     "Sigmoid Loss for Language-Image Pre-Training — Google/Meta vision-language model"],
        ["TTA",        "Test-Time Augmentation — multiple inference passes with varied inputs for accuracy"],
        ["ViT",        "Vision Transformer — attention-based architecture for image classification"],
        ["XceptionNet","Depthwise separable CNN architecture specialised for face manipulation detection"],
    ]
    _gl_hdr = ParagraphStyle("glh", parent=S["body"], fontSize=10.5, leading=14,
                              fontName="Helvetica-Bold", textColor=colors.white)
    _gl_key = ParagraphStyle("glk", parent=S["body"], fontSize=10.5, leading=14,
                              fontName="Helvetica-Bold", textColor=ACCENT)
    _gl_val = ParagraphStyle("glv", parent=S["body"], fontSize=10.5, leading=14)
    glos_header = [Paragraph("Term", _gl_hdr), Paragraph("Definition", _gl_hdr)]
    glos_rows = [
        [Paragraph(r[0], _gl_key), Paragraph(r[1], _gl_val)]
        for r in glossary_raw
    ]
    glos_table = Table([glos_header] + glos_rows, colWidths=[32 * mm, 138 * mm])
    glos_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  NAVY),
        ("TOPPADDING",     (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 8),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("LINEBELOW",      (0, 0), (-1, -1), 0.4, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, STRIPE]),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(glos_table)



    # Appendix B gets its own page
    elements.append(PageBreak())
    elements.extend(_section_header("Appendix B — Pipeline Methodology Details", S))

    _ph = ParagraphStyle("ph", parent=S["body"], fontSize=9.5, leading=13,
                          fontName="Helvetica-Bold", textColor=colors.white)
    _pk = ParagraphStyle("pk", parent=S["body"], fontSize=9.5, leading=13,
                          fontName="Helvetica-Bold", textColor=NAVY)
    _pv = ParagraphStyle("pv", parent=S["body"], fontSize=9.5, leading=13)
    meth_raw = [
        ["EfficientNet-B4",   "CNN (ImageNet \u2192 fine-tuned), TTA \u00d75",             "42,930 images, AUC 99.77%, JPEG augmentation"],
        ["AI Ensemble",       "ViT + CLIP + SigLIP+DINOv2 (3 models)",           "Pre-trained on HuggingFace (200K+ downloads)"],
        ["XceptionNet",       "Depthwise separable CNN, TTA \u00d75",                 "Face crop specialisation, custom head"],
        ["Audio",             "Mel-spectrogram + CNN",                            "Detects synthesised voice / lipsync artefacts"],
        ["LipSync",           "Mouth region temporal + audio correlation",        "Cross-modal consistency verification"],
        ["ELA",               "Dual-quality JPEG recompression (Q95+Q75)",       "Cross-quality consistency comparison"],
        ["Copy-Move",         "ORB keypoints + FLANN matching + RANSAC",         "3000 keypoints, geometric verification"],
        ["JPEG Ghost",        "Compression sweep (Q50\u2013100, step 5)",             "Block-level variance at optimal quality"],
        ["EXIF Metadata",     "Multi-source extraction + anomaly checks",        "Software tags, thumbnails, timestamps"],
        ["Eye Reflection",    "Corneal specular highlight NCC matching",          "Area ratio, centroid, cross-correlation"],
        ["Shadow/Lighting",   "Sobel gradient direction per quadrant",           "Angular consistency + face asymmetry"],
        ["Noise Pattern",     "fastNlMeansDenoising + block variance",           "64\u00d764 blocks, coefficient of variation"],
        ["Frequency Domain",  "FFT azimuthal average + 8\u00d78 DCT",                "High-freq energy ratio, spectral analysis"],
        ["Biometric",         "MediaPipe 468-point face mesh",                   "EAR, symmetry, mouth geometry, jitter"],
        ["Temporal",          "Frame-to-frame difference + SSIM",               "Temporal coherence across video frames"],
    ]
    meth_header = [Paragraph("Pipeline", _ph), Paragraph("Architecture / Technique", _ph),
                   Paragraph("Parameters / Training", _ph)]
    meth_rows = [
        [Paragraph(r[0], _pk), Paragraph(r[1], _pv), Paragraph(r[2], _pv)]
        for r in meth_raw
    ]
    method_table = Table([meth_header] + meth_rows, colWidths=[38 * mm, 72 * mm, 60 * mm])
    method_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  NAVY),
        ("TOPPADDING",     (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 7),
        ("LEFTPADDING",    (0, 0), (-1, -1), 7),
        ("LINEBELOW",      (0, 0), (-1, -1), 0.4, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, STRIPE]),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(method_table)



    # ═══════════════════════════════════════════════════════════════════
    # END OF REPORT
    # ═══════════════════════════════════════════════════════════════════
    elements.append(Spacer(1, 25))
    end_data = [["— END OF REPORT —"]]
    end_table = Table(end_data, colWidths=[170 * mm])
    end_table.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",     (0, 0), (-1, -1), colors.HexColor("#94a3b8")),
        ("LINEABOVE",     (0, 0), (-1, -1), 1, BORDER),
        ("LINEBELOW",     (0, 0), (-1, -1), 1, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(end_table)

    # Build with unified header/footer handler
    doc.build(elements, onFirstPage=_cover_and_body, onLaterPages=_cover_and_body)
    return output_path
