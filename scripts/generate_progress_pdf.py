#!/usr/bin/env python3
"""Generate a short PDF progress brief for the project."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build_pdf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#16324f"),
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#556677"),
        spaceAfter=8,
    )
    heading = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        textColor=colors.HexColor("#16324f"),
        spaceBefore=5,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        spaceAfter=3,
    )
    bullet = ParagraphStyle(
        "BulletCustom",
        parent=body,
        leftIndent=12,
        bulletIndent=0,
    )

    story = [
        Paragraph("Mammography Classification Research: Current Progress", title),
        Paragraph(
            "Short project brief generated from the current supervised-first state of the repo.",
            subtitle,
        ),
        Spacer(1, 2),
        Paragraph("What This Project Is", heading),
        Paragraph(
            "This project is a mammography classification research effort built on the CBIS-DDSM dataset. "
            "The practical goal is to build a reproducible, clinically aware foundation for breast-imaging "
            "decision support rather than optimize only for a single leaderboard metric.",
            body,
        ),
        Paragraph("Where It Started", heading),
        Paragraph(
            "The original question was whether semi-supervised learning could help in a low-label medical-imaging "
            "setting. The early work compared supervised learning, FixMatch, and Mean Teacher across multiple "
            "label budgets.",
            body,
        ),
        Paragraph("How We Got Here", heading),
        Paragraph(
            "The turning point was a baseline correction: the original supervised baseline had been artificially "
            "weakened by freeze behavior. Once corrected, supervised fine-tuning clearly beat the vanilla SSL "
            "methods explored in the repository. From there, the project moved into a supervised-first phase with "
            "controlled sweeps over resolution, backbone, regularization, and optimizer, followed by clinical-style "
            "evaluation using calibration, fixed-sensitivity specificity, and grouped exam metrics.",
            body,
        ),
        Paragraph("Current Best Result", heading),
        Paragraph(
            "The promoted baseline is <b>default_nofreeze_ls_adamw</b>: EfficientNet-B0 at 512x512 with "
            "label smoothing = 0.1 and AdamW.",
            body,
        ),
    ]

    metrics_table = Table(
        [
            ["Metric", "Value"],
            ["Mean validation ROC AUC", "0.8633"],
            ["Mean test ROC AUC", "0.7589"],
            ["Mean test PR AUC", "0.6748"],
            ["Mean test sensitivity", "0.7110"],
            ["Mean test specificity", "0.6651"],
            ["Mean specificity at 0.90 target sensitivity", "0.5833"],
            ["Mean exam-level ROC AUC", "0.7668"],
        ],
        colWidths=[95 * mm, 42 * mm],
        hAlign="LEFT",
    )
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#edf3f8")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9fb4c7")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.extend([metrics_table, Spacer(1, 5)])

    story.extend(
        [
            Paragraph("Results Summary", heading),
            Paragraph(
                "The project started as a supervised-vs-SSL comparison, but the corrected supervised baseline "
                "outperformed the vanilla FixMatch and Mean Teacher implementations. Later supervised sweeps showed "
                "that the best regime was not a larger backbone or larger image size, but a regularized 512px "
                "EfficientNet-B0. The final head-to-head then promoted AdamW over Adam for that family.",
                body,
            ),
            Paragraph("Future Development", heading),
            Paragraph(
                "• Prioritize calibration work, starting with temperature scaling.", bullet),
            Paragraph(
                "• Expand failure analysis for false positives, false negatives, and subgroup behavior.", bullet),
            Paragraph(
                "• Move from single-image classification toward exam-level and multi-view modeling.", bullet),
            Paragraph(
                "• Validate on additional mammography datasets to test source-shift robustness.", bullet),
            Paragraph("Possible Clinical Implementation", heading),
            Paragraph(
                "A realistic clinical path would frame the model as radiologist decision support rather than "
                "autonomous diagnosis. Before any serious deployment discussion, the system would need stronger "
                "calibration, external validation, subgroup robustness evidence, reader-study style evaluation, "
                "and traceable model documentation with change control.",
                body,
            ),
            Paragraph(
                "The current state is best described as a strong, experimentally grounded research baseline that is "
                "now ready for calibration work, deeper analysis, and clinically oriented validation.",
                body,
            ),
        ]
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="Mammography Classification Research: Current Progress",
        author="Codex",
    )
    doc.build(story)


def main() -> None:
    build_pdf(Path("output/pdf/current_progress_brief.pdf"))


if __name__ == "__main__":
    main()
