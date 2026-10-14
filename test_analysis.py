"""
MalGuard-XAI Stage 3 Tests
Tests for SHAP Explainer, Claude Analyzer, IOC Extractor, PDF Report Generator,
and the unified analyze pipeline.
"""

import os
import sys
import json
import tempfile
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def feature_names():
    """Returns the sorted feature column names from the test dataset."""
    test_csv = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    if os.path.exists(test_csv):
        df = pd.read_csv(test_csv, nrows=1)
        return sorted([c for c in df.columns if c != "label"])
    else:
        # Minimal fallback feature set for testing
        return sorted([
            "file_size", "file_entropy", "header_number_of_sections",
            "sec_entropy_max", "sec_entropy_mean", "sec_entropy_min",
            "sec_executable_count", "sec_writable_count",
            "imports_dll_count", "imports_func_count", "exports_count",
            "import_VirtualAlloc", "import_WriteProcessMemory",
            "import_CreateRemoteThread", "import_GetProcAddress",
            "yara_suspicious_apis", "yara_anti_debugging",
            "yara_common_packers", "yara_suspicious_network_strings",
            "opt_address_of_entry_point", "opt_image_base",
            "opt_check_sum", "opt_size_of_code",
            "opcode_freq_mov", "opcode_freq_push", "opcode_freq_xor",
            "opcode_total_count"
        ])


@pytest.fixture
def sample_malicious_features(feature_names):
    """Creates a sample feature vector representing a malicious PE file."""
    features = {name: 0.0 for name in feature_names}
    features.update({
        "file_size": 245760,
        "file_entropy": 7.85,
        "header_number_of_sections": 6,
        "sec_entropy_max": 7.92,
        "sec_entropy_mean": 6.5,
        "sec_entropy_min": 3.2,
        "sec_executable_count": 2,
        "sec_writable_count": 3,
        "imports_dll_count": 4,
        "imports_func_count": 42,
        "exports_count": 0,
        "import_VirtualAlloc": 1,
        "import_WriteProcessMemory": 1,
        "import_CreateRemoteThread": 1,
        "import_GetProcAddress": 1,
        "yara_suspicious_apis": 1,
        "yara_anti_debugging": 1,
        "yara_common_packers": 0,
        "yara_suspicious_network_strings": 1,
        "opt_address_of_entry_point": 0x1000,
        "opt_image_base": 0x400000,
        "opt_check_sum": 0,
        "opt_size_of_code": 102400,
    })
    return features


@pytest.fixture
def sample_benign_features(feature_names):
    """Creates a sample feature vector representing a benign PE file."""
    features = {name: 0.0 for name in feature_names}
    features.update({
        "file_size": 524288,
        "file_entropy": 5.2,
        "header_number_of_sections": 5,
        "sec_entropy_max": 6.1,
        "sec_entropy_mean": 4.8,
        "sec_entropy_min": 2.0,
        "sec_executable_count": 1,
        "sec_writable_count": 1,
        "imports_dll_count": 8,
        "imports_func_count": 120,
        "exports_count": 5,
        "import_VirtualAlloc": 0,
        "import_WriteProcessMemory": 0,
        "import_CreateRemoteThread": 0,
        "import_GetProcAddress": 1,
        "yara_suspicious_apis": 0,
        "yara_anti_debugging": 0,
        "yara_common_packers": 0,
        "yara_suspicious_network_strings": 0,
        "opt_address_of_entry_point": 0x1000,
        "opt_image_base": 0x400000,
        "opt_check_sum": 12345,
        "opt_size_of_code": 204800,
    })
    return features


@pytest.fixture
def dummy_binary(tmp_path):
    """Creates a small dummy binary file for testing IOC hash extraction."""
    dummy = tmp_path / "test_sample.exe"
    dummy.write_bytes(b"MZ" + os.urandom(512))
    return str(dummy)


@pytest.fixture
def reports_dir(tmp_path):
    """Creates a temporary reports directory."""
    rdir = tmp_path / "reports"
    rdir.mkdir()
    return str(rdir)


# ─────────────────────────────────────────────
# Test: SHAPExplainer
# ─────────────────────────────────────────────

class TestSHAPExplainer:
    """Tests for the SHAP Explainability module."""

    def test_explain_sample_heuristic_fallback(self, feature_names, sample_malicious_features, reports_dir):
        """Test that SHAP explainer falls back to heuristic mode when no model is provided."""
        from src.analysis.shap_explainer import SHAPExplainer

        # Create a mock model that triggers fallback
        class MockModel:
            pass

        explainer = SHAPExplainer(model=MockModel(), feature_names=feature_names)
        # Force fallback
        explainer.explainer = None

        X = pd.DataFrame([sample_malicious_features], columns=feature_names)
        attributions = explainer.explain_sample(X)

        assert isinstance(attributions, list)
        assert len(attributions) <= 10
        assert all(isinstance(a, tuple) and len(a) == 3 for a in attributions)

    def test_explain_sample_with_trained_model(self, feature_names, sample_malicious_features, reports_dir):
        """Test SHAP explainer with a real trained model if available."""
        from src.analysis.shap_explainer import SHAPExplainer

        model_path = os.path.join(PROJECT_ROOT, "models", "rf_model.pkl")
        if not os.path.exists(model_path):
            pytest.skip("Trained RF model not found — skipping SHAP integration test")

        import joblib
        rf_model = joblib.load(model_path)

        explainer = SHAPExplainer(model=rf_model, feature_names=feature_names)
        X = pd.DataFrame([sample_malicious_features], columns=feature_names)
        attributions = explainer.explain_sample(X)

        assert isinstance(attributions, list)
        assert len(attributions) > 0
        assert len(attributions) <= 10

    def test_plot_attributions(self, feature_names, sample_malicious_features, reports_dir):
        """Test that plot_attributions generates a PNG file."""
        from src.analysis.shap_explainer import SHAPExplainer

        class MockModel:
            pass

        explainer = SHAPExplainer(model=MockModel(), feature_names=feature_names)
        explainer.explainer = None

        X = pd.DataFrame([sample_malicious_features], columns=feature_names)
        attributions = explainer.explain_sample(X)
        chart_path = explainer.plot_attributions(attributions, out_dir=reports_dir)

        assert os.path.exists(chart_path)
        assert chart_path.endswith(".png")
        assert os.path.getsize(chart_path) > 0


# ─────────────────────────────────────────────
# Test: ClaudeAnalyzer
# ─────────────────────────────────────────────

class TestClaudeAnalyzer:
    """Tests for the Claude AI Analyzer module."""

    def test_mock_fallback_malicious(self, sample_malicious_features):
        """Test that the mock fallback engine produces a structured malicious analysis."""
        from src.analysis.claude_analyzer import ClaudeAnalyzer

        analyzer = ClaudeAnalyzer(api_key="your_key_here")  # Forces mock mode
        result = analyzer.analyze_threat(
            features=sample_malicious_features,
            verdict="Malicious",
            confidence=0.92
        )

        assert isinstance(result, dict)
        assert result["verdict"] == "Malicious"
        assert result["confidence"] == 0.92
        assert result["risk_level"] in ("Critical", "High", "Medium", "Low")
        assert "summary" in result
        assert "detailed_analysis" in result
        assert "remediation" in result
        assert len(result["detailed_analysis"]) > 20

    def test_mock_fallback_benign(self, sample_benign_features):
        """Test that the mock fallback engine produces a structured benign analysis."""
        from src.analysis.claude_analyzer import ClaudeAnalyzer

        analyzer = ClaudeAnalyzer(api_key="your_key_here")
        result = analyzer.analyze_threat(
            features=sample_benign_features,
            verdict="Benign",
            confidence=0.88
        )

        assert isinstance(result, dict)
        assert result["verdict"] == "Benign"
        assert result["risk_level"] == "Low"
        assert "summary" in result
        assert "detailed_analysis" in result

    def test_client_not_initialized_without_key(self):
        """Test that the Anthropic client is NOT initialized with placeholder keys."""
        from src.analysis.claude_analyzer import ClaudeAnalyzer

        analyzer = ClaudeAnalyzer(api_key="your_key_here")
        assert analyzer.client is None

    def test_structured_output_keys(self, sample_malicious_features):
        """Test that all expected keys are present in the output."""
        from src.analysis.claude_analyzer import ClaudeAnalyzer

        analyzer = ClaudeAnalyzer(api_key="")
        result = analyzer.analyze_threat(
            features=sample_malicious_features,
            verdict="Malicious",
            confidence=0.95
        )

        required_keys = {"verdict", "confidence", "risk_level", "summary", "detailed_analysis", "remediation"}
        assert required_keys.issubset(result.keys())


# ─────────────────────────────────────────────
# Test: IOCExtractor
# ─────────────────────────────────────────────

class TestIOCExtractor:
    """Tests for the Indicators of Compromise Extractor module."""

    def test_get_file_hashes(self, dummy_binary):
        """Test that MD5 and SHA-256 hashes are computed correctly."""
        from src.analysis.ioc_extractor import IOCExtractor

        extractor = IOCExtractor()
        md5, sha256 = extractor.get_file_hashes(dummy_binary)

        assert isinstance(md5, str)
        assert isinstance(sha256, str)
        assert len(md5) == 32
        assert len(sha256) == 64

    def test_get_file_hashes_missing_file(self):
        """Test graceful handling of missing files."""
        from src.analysis.ioc_extractor import IOCExtractor

        extractor = IOCExtractor()
        md5, sha256 = extractor.get_file_hashes("/nonexistent/file.exe")
        assert md5 == "n/a"
        assert sha256 == "n/a"

    def test_extract_iocs_with_malicious_features(self, dummy_binary, sample_malicious_features):
        """Test IOC extraction from a malicious feature set."""
        from src.analysis.ioc_extractor import IOCExtractor

        extractor = IOCExtractor()
        iocs = extractor.extract_iocs(dummy_binary, sample_malicious_features)

        assert isinstance(iocs, dict)
        assert "md5" in iocs
        assert "sha256" in iocs
        assert "suspicious_apis" in iocs
        assert isinstance(iocs["suspicious_apis"], list)
        assert "network_iocs" in iocs
        assert "registry_iocs" in iocs
        assert iocs["total_iocs_count"] >= 0

    def test_extract_iocs_with_benign_features(self, dummy_binary, sample_benign_features):
        """Test that benign features produce fewer IOCs."""
        from src.analysis.ioc_extractor import IOCExtractor

        extractor = IOCExtractor()
        iocs = extractor.extract_iocs(dummy_binary, sample_benign_features)

        assert isinstance(iocs, dict)
        # Benign should have fewer suspicious indicators
        assert len(iocs.get("network_iocs", [])) == 0
        assert len(iocs.get("registry_iocs", [])) == 0


# ─────────────────────────────────────────────
# Test: PDFReportGenerator
# ─────────────────────────────────────────────

class TestPDFReportGenerator:
    """Tests for the PDF threat report generator."""

    def test_generate_malicious_report(self, reports_dir, sample_malicious_features):
        """Test that a complete PDF is generated for a malicious verdict."""
        from src.analysis.report_generator import PDFReportGenerator

        generator = PDFReportGenerator(reports_dir=reports_dir)

        claude_analysis = {
            "verdict": "Malicious",
            "confidence": 0.92,
            "risk_level": "Critical",
            "summary": "Highly suspicious packed executable with anti-debugging capabilities.",
            "detailed_analysis": "The ML ensemble classified this binary as MALICIOUS with 92% confidence. "
                                 "High entropy values and YARA signature matches confirm obfuscation.",
            "remediation": "Quarantine immediately. Run dynamic sandbox analysis."
        }

        ioc_data = {
            "filename": "test_malware.exe",
            "md5": "d41d8cd98f00b204e9800998ecf8427e",
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "suspicious_apis": ["VirtualAlloc", "WriteProcessMemory", "CreateRemoteThread"],
            "network_iocs": ["http://malicious-command-control.xyz/gate.php"],
            "registry_iocs": [r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run\MalGuardUpdate"],
            "total_iocs_count": 5
        }

        attributions = [
            ("file_entropy", 0.35, 7.85),
            ("sec_entropy_max", 0.30, 7.92),
            ("import_VirtualAlloc", 0.25, 1.0),
            ("yara_suspicious_apis", 0.40, 1.0),
            ("import_WriteProcessMemory", 0.22, 1.0),
        ]

        pdf_path = generator.generate_report(
            filename="test_malware.exe",
            verdict="Malicious",
            confidence=0.92,
            features=sample_malicious_features,
            claude_analysis=claude_analysis,
            ioc_data=ioc_data,
            shap_chart_path=None,
            attributions=attributions
        )

        assert os.path.exists(pdf_path)
        assert pdf_path.endswith(".pdf")
        assert os.path.getsize(pdf_path) > 1000  # Reasonable minimum size for a multi-page PDF

    def test_generate_benign_report(self, reports_dir, sample_benign_features):
        """Test PDF generation for a benign verdict."""
        from src.analysis.report_generator import PDFReportGenerator

        generator = PDFReportGenerator(reports_dir=reports_dir)

        claude_analysis = {
            "verdict": "Benign",
            "confidence": 0.88,
            "risk_level": "Low",
            "summary": "File appears to be a legitimate executable.",
            "detailed_analysis": "Standard compilation profile detected.",
            "remediation": "No action required."
        }

        ioc_data = {
            "filename": "notepad.exe",
            "md5": "abc123",
            "sha256": "def456",
            "suspicious_apis": [],
            "network_iocs": [],
            "registry_iocs": [],
            "total_iocs_count": 0
        }

        pdf_path = generator.generate_report(
            filename="notepad.exe",
            verdict="Benign",
            confidence=0.88,
            features=sample_benign_features,
            claude_analysis=claude_analysis,
            ioc_data=ioc_data
        )

        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 500

    def test_report_with_shap_chart(self, reports_dir, feature_names, sample_malicious_features):
        """Test PDF generation with an embedded SHAP chart image."""
        from src.analysis.shap_explainer import SHAPExplainer
        from src.analysis.report_generator import PDFReportGenerator

        # Generate a SHAP chart first
        class MockModel:
            pass

        explainer = SHAPExplainer(model=MockModel(), feature_names=feature_names)
        explainer.explainer = None
        X = pd.DataFrame([sample_malicious_features], columns=feature_names)
        attributions = explainer.explain_sample(X)
        chart_path = explainer.plot_attributions(attributions, out_dir=reports_dir)

        # Generate PDF with chart embedded
        generator = PDFReportGenerator(reports_dir=reports_dir)
        pdf_path = generator.generate_report(
            filename="chart_test.exe",
            verdict="Malicious",
            confidence=0.95,
            features=sample_malicious_features,
            claude_analysis={
                "verdict": "Malicious", "confidence": 0.95, "risk_level": "High",
                "summary": "Test.", "detailed_analysis": "Test analysis.", "remediation": "Test."
            },
            ioc_data={
                "filename": "chart_test.exe", "md5": "abc", "sha256": "def",
                "suspicious_apis": [], "network_iocs": [], "registry_iocs": [],
                "total_iocs_count": 0
            },
            shap_chart_path=chart_path,
            attributions=attributions
        )

        assert os.path.exists(pdf_path)
        # PDF with embedded image should be larger
        assert os.path.getsize(pdf_path) > 2000


# ─────────────────────────────────────────────
# Test: End-to-End Integration (with mock model)
# ─────────────────────────────────────────────

class TestEndToEndIntegration:
    """Integration tests verifying the full analysis pipeline works together."""

    def test_full_pipeline_with_synthetic_features(
        self, feature_names, sample_malicious_features, dummy_binary, reports_dir
    ):
        """Test the complete analysis pipeline using mock/fallback modes."""
        from src.analysis.shap_explainer import SHAPExplainer
        from src.analysis.claude_analyzer import ClaudeAnalyzer
        from src.analysis.ioc_extractor import IOCExtractor
        from src.analysis.report_generator import PDFReportGenerator

        # Step 1: SHAP (heuristic fallback)
        class MockModel:
            pass

        explainer = SHAPExplainer(model=MockModel(), feature_names=feature_names)
        explainer.explainer = None
        X = pd.DataFrame([sample_malicious_features], columns=feature_names)
        attributions = explainer.explain_sample(X)
        chart_path = explainer.plot_attributions(attributions, out_dir=reports_dir)

        assert os.path.exists(chart_path)
        assert len(attributions) > 0

        # Step 2: Claude AI (mock fallback)
        claude = ClaudeAnalyzer(api_key="")
        analysis = claude.analyze_threat(
            features=sample_malicious_features,
            verdict="Malicious",
            confidence=0.90
        )
        assert "verdict" in analysis
        assert "risk_level" in analysis

        # Step 3: IOC extraction
        ioc_ext = IOCExtractor()
        iocs = ioc_ext.extract_iocs(dummy_binary, sample_malicious_features)
        assert "md5" in iocs
        assert iocs["total_iocs_count"] >= 0

        # Step 4: PDF report
        pdf_gen = PDFReportGenerator(reports_dir=reports_dir)
        pdf_path = pdf_gen.generate_report(
            filename=os.path.basename(dummy_binary),
            verdict="Malicious",
            confidence=0.90,
            features=sample_malicious_features,
            claude_analysis=analysis,
            ioc_data=iocs,
            shap_chart_path=chart_path,
            attributions=attributions
        )

        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 2000
        print(f"\n  ✔ Integration test PDF: {pdf_path}")
