import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """
    Generates professional multi-page threat analysis PDF reports using ReportLab.
    Integrates ML classification results, SHAP explanations, Claude AI analysis,
    and Indicators of Compromise into a single, downloadable document.
    """

    # Color palette
    COLOR_MALICIOUS = colors.HexColor("#CC0000")
    COLOR_BENIGN = colors.HexColor("#228B22")
    COLOR_HEADER_BG = colors.HexColor("#1A1A2E")
    COLOR_HEADER_TEXT = colors.white
    COLOR_ACCENT = colors.HexColor("#0F3460")
    COLOR_TABLE_HEADER = colors.HexColor("#16213E")
    COLOR_TABLE_ALT = colors.HexColor("#F0F0F0")
    COLOR_BORDER = colors.HexColor("#CCCCCC")

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._register_custom_styles()

    def _register_custom_styles(self):
        """Registers custom paragraph styles for the report."""
        self.styles.add(ParagraphStyle(
            name="ReportTitle",
            parent=self.styles["Heading1"],
            fontSize=22,
            leading=28,
            textColor=self.COLOR_HEADER_BG,
            alignment=TA_CENTER,
            spaceAfter=6
        ))
        self.styles.add(ParagraphStyle(
            name="SectionTitle",
            parent=self.styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=self.COLOR_ACCENT,
            spaceBefore=16,
            spaceAfter=8,
            borderWidth=1,
            borderColor=self.COLOR_ACCENT,
            borderPadding=(0, 0, 4, 0)
        ))
        self.styles.add(ParagraphStyle(
            name="BodyJustified",
            parent=self.styles["BodyText"],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        ))
        self.styles.add(ParagraphStyle(
            name="VerdictMalicious",
            parent=self.styles["Heading1"],
            fontSize=18,
            leading=24,
            textColor=self.COLOR_MALICIOUS,
            alignment=TA_CENTER
        ))
        self.styles.add(ParagraphStyle(
            name="VerdictBenign",
            parent=self.styles["Heading1"],
            fontSize=18,
            leading=24,
            textColor=self.COLOR_BENIGN,
            alignment=TA_CENTER
        ))
        self.styles.add(ParagraphStyle(
            name="SmallFooter",
            parent=self.styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.gray,
            alignment=TA_CENTER
        ))
        self.styles.add(ParagraphStyle(
            name="CellText",
            parent=self.styles["Normal"],
            fontSize=9,
            leading=12,
            wordWrap="CJK"
        ))

    def generate_report(
        self,
        filename: str,
        verdict: str,
        confidence: float,
        features: Dict[str, Any],
        claude_analysis: Dict[str, Any],
        ioc_data: Dict[str, Any],
        shap_chart_path: Optional[str] = None,
        attributions: Optional[List] = None
    ) -> str:
        """
        Generates the full threat analysis PDF report.

        Args:
            filename: Name of the analyzed PE file.
            verdict: 'Malicious' or 'Benign'.
            confidence: ML model confidence score (0.0 - 1.0).
            features: Extracted PE feature dictionary.
            claude_analysis: Dict with keys: verdict, confidence, risk_level, summary, detailed_analysis, remediation.
            ioc_data: Dict with keys: md5, sha256, suspicious_apis, network_iocs, registry_iocs.
            shap_chart_path: Absolute path to the saved SHAP bar chart image.
            attributions: List of (feature_name, shap_value, actual_value) tuples.

        Returns:
            Path to the generated PDF file.
        """
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"MalGuard_Report_{safe_name}_{timestamp}.pdf"
        pdf_path = os.path.join(self.reports_dir, pdf_filename)

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
            title=f"MalGuard-XAI Threat Report: {filename}",
            author="MalGuard-XAI Automated Analyzer"
        )

        elements = []

        # ── Page 1: Executive Summary ──
        elements.append(Paragraph("MalGuard-XAI", self.styles["ReportTitle"]))
        elements.append(Paragraph("Automated Threat Analysis Report", self.styles["Normal"]))
        elements.append(Spacer(1, 6))
        elements.append(HRFlowable(
            width="100%", thickness=2, color=self.COLOR_ACCENT, spaceAfter=12
        ))

        # Verdict banner
        is_malicious = verdict.lower() == "malicious"
        verdict_style = "VerdictMalicious" if is_malicious else "VerdictBenign"
        verdict_text = f"⚠ VERDICT: {verdict.upper()} ⚠" if is_malicious else f"✔ VERDICT: {verdict.upper()} ✔"
        elements.append(Paragraph(verdict_text, self.styles[verdict_style]))
        elements.append(Spacer(1, 4))

        # Summary metadata table
        risk_level = claude_analysis.get("risk_level", "Unknown")
        risk_color = self.COLOR_MALICIOUS if risk_level in ("Critical", "High") else self.COLOR_BENIGN
        meta_data = [
            ["File Name", filename],
            ["Analysis Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["ML Confidence", f"{confidence * 100:.1f}%"],
            ["Risk Level", risk_level],
            ["Malware Family", claude_analysis.get("malware_family", "None")],
            ["MD5 Hash", ioc_data.get("md5", "N/A")],
            ["SHA-256 Hash", ioc_data.get("sha256", "N/A")[:48] + "..."],
        ]
        meta_table = Table(meta_data, colWidths=[2.0 * inch, 4.5 * inch])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), self.COLOR_TABLE_HEADER),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, self.COLOR_BORDER),
            ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, self.COLOR_TABLE_ALT]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 12))

        # Executive Summary text
        elements.append(Paragraph("1. Executive Summary", self.styles["SectionTitle"]))
        summary_text = claude_analysis.get("summary", "No summary available.")
        elements.append(Paragraph(summary_text, self.styles["BodyJustified"]))

        # ── Page 2: Static Analysis Details ──
        elements.append(Paragraph("2. Static Metadata Analysis", self.styles["SectionTitle"]))

        static_rows = [
            ["Attribute", "Value"],
            ["File Size (bytes)", str(features.get("file_size", 0))],
            ["File Entropy", f"{features.get('file_entropy', 0.0):.4f}"],
            ["Number of Sections", str(features.get("header_number_of_sections", 0))],
            ["Max Section Entropy", f"{features.get('sec_entropy_max', 0.0):.4f}"],
            ["Mean Section Entropy", f"{features.get('sec_entropy_mean', 0.0):.4f}"],
            ["Executable Sections", str(features.get("sec_executable_count", 0))],
            ["Writable Sections", str(features.get("sec_writable_count", 0))],
            ["Imported DLLs", str(features.get("imports_dll_count", 0))],
            ["Imported Functions", str(features.get("imports_func_count", 0))],
            ["Exported Functions", str(features.get("exports_count", 0))],
            ["Entry Point Address", hex(int(features.get("opt_address_of_entry_point", 0)))],
            ["Image Base", hex(int(features.get("opt_image_base", 0)))],
            ["Checksum", str(int(features.get("opt_check_sum", 0)))],
        ]
        static_table = Table(static_rows, colWidths=[3.0 * inch, 3.5 * inch])
        static_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.COLOR_TABLE_HEADER),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, self.COLOR_BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, self.COLOR_TABLE_ALT]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(static_table)
        elements.append(Spacer(1, 12))

        # ── Section 3: Explainable AI (XAI) ──
        elements.append(Paragraph("3. Explainable AI (XAI) — Feature Attributions", self.styles["SectionTitle"]))

        if shap_chart_path and os.path.exists(shap_chart_path):
            elements.append(Paragraph(
                "The chart below shows the top features that drove the ML prediction. "
                "Red bars push the prediction toward <b>Malicious</b>; blue bars push toward <b>Benign</b>.",
                self.styles["BodyJustified"]
            ))
            elements.append(Spacer(1, 6))
            try:
                shap_img = Image(shap_chart_path, width=6.0 * inch, height=3.3 * inch)
                elements.append(shap_img)
            except Exception as e:
                logger.warning(f"Failed to embed SHAP chart: {e}")
                elements.append(Paragraph(
                    f"[SHAP chart could not be loaded: {e}]", self.styles["BodyJustified"]
                ))
        elif attributions:
            # Fallback: render attributions as a table
            attr_rows = [["Feature", "Contribution", "Actual Value"]]
            for name, val, actual in attributions:
                direction = "▲ Malicious" if val >= 0 else "▼ Benign"
                attr_rows.append([name, f"{val:+.4f} ({direction})", f"{actual:.4f}"])

            attr_table = Table(attr_rows, colWidths=[2.5 * inch, 2.0 * inch, 2.0 * inch])
            attr_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), self.COLOR_TABLE_HEADER),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, self.COLOR_BORDER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, self.COLOR_TABLE_ALT]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(attr_table)
        else:
            elements.append(Paragraph(
                "[No SHAP data available for this analysis.]", self.styles["BodyJustified"]
            ))
        elements.append(Spacer(1, 12))

        # ── Section 4: Threat Intelligence Reasoning ──
        elements.append(Paragraph("4. Threat Intelligence Reasoning", self.styles["SectionTitle"]))
        detailed = claude_analysis.get("detailed_analysis", "No detailed analysis available.")
        elements.append(Paragraph(detailed, self.styles["BodyJustified"]))
        elements.append(Spacer(1, 6))

        remediation = claude_analysis.get("remediation", "")
        if remediation:
            elements.append(Paragraph("<b>Recommended Remediation:</b>", self.styles["BodyJustified"]))
            elements.append(Paragraph(remediation, self.styles["BodyJustified"]))
        elements.append(Spacer(1, 12))

        # ── Section 5: Indicators of Compromise (IOCs) ──
        elements.append(Paragraph("5. Indicators of Compromise (IOCs)", self.styles["SectionTitle"]))

        ioc_rows = [["Category", "Indicator"]]

        # File hashes
        ioc_rows.append(["MD5 Hash", ioc_data.get("md5", "N/A")])
        ioc_rows.append(["SHA-256 Hash", ioc_data.get("sha256", "N/A")])

        # Suspicious APIs
        apis = ioc_data.get("suspicious_apis", [])
        if apis:
            for api in apis:
                ioc_rows.append(["Suspicious API", api])
        else:
            ioc_rows.append(["Suspicious API", "None detected"])

        # Network IOCs
        net_iocs = ioc_data.get("network_iocs", [])
        if net_iocs:
            for net in net_iocs:
                ioc_rows.append(["Network Endpoint", net])
        else:
            ioc_rows.append(["Network Endpoint", "None detected"])

        # Registry IOCs
        reg_iocs = ioc_data.get("registry_iocs", [])
        if reg_iocs:
            for reg in reg_iocs:
                ioc_rows.append(["Registry Key", Paragraph(reg, self.styles["CellText"])])
        else:
            ioc_rows.append(["Registry Key", "None detected"])

        ioc_table = Table(ioc_rows, colWidths=[2.0 * inch, 4.5 * inch])
        ioc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.COLOR_TABLE_HEADER),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, self.COLOR_BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, self.COLOR_TABLE_ALT]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(ioc_table)
        elements.append(Spacer(1, 20))

        # Footer
        elements.append(HRFlowable(
            width="100%", thickness=1, color=self.COLOR_BORDER, spaceAfter=6
        ))
        elements.append(Paragraph(
            f"Generated by MalGuard-XAI Automated Threat Analysis Engine — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles["SmallFooter"]
        ))
        elements.append(Paragraph(
            "CONFIDENTIAL — For authorized cybersecurity personnel only.",
            self.styles["SmallFooter"]
        ))

        # Build PDF
        try:
            doc.build(elements)
            logger.info(f"PDF threat report generated: {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            raise

        return pdf_path
