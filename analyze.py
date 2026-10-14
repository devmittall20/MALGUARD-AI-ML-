"""
MalGuard-XAI Unified Analyzer Pipeline

End-to-end command-line interface that orchestrates:
1. PE Feature Extraction (Stage 1)
2. ML Ensemble Classification (Stage 2)
3. SHAP Explainability, Claude AI Reasoning, IOC Extraction, and PDF Report Generation (Stage 3)

Usage:
    python -m src.analysis.analyze --file <path_to_binary>
    python -m src.analysis.analyze --file <path_to_binary> --models-dir models --reports-dir reports
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

import pandas as pd
import numpy as np

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# MalGuard-XAI internal imports
from ..extraction.extractor import PEFeatureExtractor
from ..models.ensemble import PEVotingEnsemble
from .shap_explainer import SHAPExplainer
from .claude_analyzer import ClaudeAnalyzer
from .ioc_extractor import IOCExtractor
from .report_generator import PDFReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("malguard_xai.analyzer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MalGuard-XAI: End-to-End PE File Threat Analyzer"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        required=True,
        help="Path to the PE executable file to analyze"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained model files (default: models)"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory to save generated reports (default: reports)"
    )
    parser.add_argument(
        "--yara-rules",
        type=str,
        default=None,
        help="Path to custom YARA rules file (default: src/extraction/rules.yar)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data (used to load feature schema)"
    )
    return parser.parse_args()


def get_feature_names(data_dir: str) -> list:
    """
    Loads feature column names from the training dataset to ensure schema consistency.
    Falls back to a hardcoded list if the data directory is not available.
    """
    train_path = os.path.join(data_dir, "train.csv")
    if os.path.exists(train_path):
        df = pd.read_csv(train_path, nrows=1)
        feature_cols = [c for c in df.columns if c != "label"]
        return sorted(feature_cols)

    # Fallback: try test.csv
    test_path = os.path.join(data_dir, "test.csv")
    if os.path.exists(test_path):
        df = pd.read_csv(test_path, nrows=1)
        feature_cols = [c for c in df.columns if c != "label"]
        return sorted(feature_cols)

    logger.warning("Could not find training data to load feature schema. Using extractor schema.")
    return None


def features_to_dataframe(features: dict, feature_names: list) -> pd.DataFrame:
    """
    Converts a raw feature dictionary into a single-row DataFrame aligned to the model's
    expected feature schema. Missing features are filled with 0.
    """
    row = {}
    for col in feature_names:
        row[col] = features.get(col, 0)
    return pd.DataFrame([row], columns=feature_names)


def main(custom_args=None):
    args = custom_args if custom_args is not None else parse_args()

    file_path = os.path.abspath(args.file)
    models_dir = os.path.abspath(args.models_dir)
    reports_dir = os.path.abspath(args.reports_dir)
    data_dir = os.path.abspath(args.data_dir)

    # Resolve YARA rules path
    if args.yara_rules:
        yara_rules_path = os.path.abspath(args.yara_rules)
    else:
        # Default: look relative to this file's package
        yara_rules_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "extraction", "rules.yar"
        )
        if not os.path.exists(yara_rules_path):
            yara_rules_path = None

    os.makedirs(reports_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  MalGuard-XAI - Automated Threat Analysis Pipeline")
    print("=" * 70)

    # ── Step 1: Validate Input ──
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        print(f"\n[ERROR] File not found: {file_path}")
        sys.exit(1)

    filename = os.path.basename(file_path)
    print(f"\n  Target File: {filename}")
    print(f"  Full Path:   {file_path}")
    print(f"  File Size:   {os.path.getsize(file_path):,} bytes")
    print("-" * 70)

    # ── Step 2: Extract PE Features ──
    print("\n[1/5] Extracting static PE features...")
    extractor = PEFeatureExtractor(yara_rules_path=yara_rules_path)
    features = extractor.extract_features(file_path)

    if features.get("extraction_error"):
        logger.warning(f"Feature extraction encountered issues: {features.get('extraction_error')}")
        print(f"  [!] Warning: {features.get('extraction_error')}")

    print(f"  [+] Extracted {len(features)} raw features from {filename}")

    # ── Step 3: Load Model and Classify ──
    print("\n[2/5] Loading ML ensemble and classifying...")
    feature_names = get_feature_names(data_dir)
    if feature_names is None:
        # Build feature names from extractor's schema
        feature_names = sorted([k for k in features.keys() if k not in ("extraction_error", "label")])

    X_sample = features_to_dataframe(features, feature_names)

    try:
        ensemble = PEVotingEnsemble.load(os.path.join(models_dir, "ensemble_model.pkl"))
        ensemble.load_submodels(models_dir)
    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        print(f"\n[ERROR] {e}")
        print("  Run the training pipeline first: python -m src.models.train --action train-all")
        sys.exit(1)

    # Get prediction probabilities
    proba = ensemble.predict_proba(X_sample)
    malicious_prob = float(proba[0, 1])
    prediction = int(malicious_prob >= 0.5)
    verdict = "Malicious" if prediction == 1 else "Benign"
    confidence = malicious_prob if prediction == 1 else (1.0 - malicious_prob)

    print(f"  [+] Verdict: {verdict}")
    print(f"  [+] Confidence: {confidence * 100:.1f}%")
    print(f"  [+] Malicious Probability: {malicious_prob * 100:.1f}%")

    # ── Step 4: SHAP Explainability ──
    print("\n[3/5] Computing SHAP feature attributions...")

    # Use the RF sub-model for SHAP (TreeExplainer compatible) — static features only
    static_feature_names = [f for f in feature_names if not f.startswith("api_seq_") and not f.startswith("byte_")]
    X_static = X_sample[static_feature_names]
    shap_explainer = SHAPExplainer(model=ensemble.rf_model, feature_names=static_feature_names)
    attributions = shap_explainer.explain_sample(X_static)
    shap_chart_path = shap_explainer.plot_attributions(attributions, out_dir=reports_dir)

    print(f"  [+] Top contributing features:")
    for i, (name, val, actual) in enumerate(attributions[:5], 1):
        direction = "^ Malicious" if val >= 0 else "v Benign"
        print(f"     {i}. {name}: {val:+.4f} ({direction}) [value={actual:.4f}]")
    print(f"  [+] SHAP chart saved to: {shap_chart_path}")

    # ── Step 5: Claude AI Reasoning + IOC Extraction ──
    print("\n[4/5] Running AI threat reasoning & IOC extraction...")

    claude_analyzer = ClaudeAnalyzer()
    claude_result = claude_analyzer.analyze_threat(
        features=features,
        verdict=verdict,
        confidence=confidence
    )

    ioc_extractor = IOCExtractor()
    ioc_data = ioc_extractor.extract_iocs(file_path, features)
    
    # Merge Claude's AI-generated IOCs
    claude_iocs = claude_result.get("iocs", {})
    if claude_iocs:
        if "network" in claude_iocs and claude_iocs["network"]:
            ioc_data["network_iocs"] = list(set(ioc_data.get("network_iocs", []) + claude_iocs["network"]))
        if "registry" in claude_iocs and claude_iocs["registry"]:
            ioc_data["registry_iocs"] = list(set(ioc_data.get("registry_iocs", []) + claude_iocs["registry"]))
        # Update total count
        ioc_data["total_iocs_count"] = len(ioc_data.get("suspicious_apis", [])) + len(ioc_data.get("network_iocs", [])) + len(ioc_data.get("registry_iocs", []))

    print(f"  [+] Risk Level: {claude_result.get('risk_level', 'Unknown')}")
    print(f"  [+] Malware Family: {claude_result.get('malware_family', 'None')}")
    print(f"  [+] IOCs Found: {ioc_data.get('total_iocs_count', 0)}")
    if ioc_data.get("suspicious_apis"):
        print(f"  [+] Suspicious APIs: {', '.join(ioc_data['suspicious_apis'][:5])}")

    # ── Step 6: Generate PDF Report ──
    print("\n[5/5] Generating PDF threat analysis report...")

    pdf_generator = PDFReportGenerator(reports_dir=reports_dir)
    pdf_path = pdf_generator.generate_report(
        filename=filename,
        verdict=verdict,
        confidence=confidence,
        features=features,
        claude_analysis=claude_result,
        ioc_data=ioc_data,
        shap_chart_path=shap_chart_path,
        attributions=attributions
    )

    print(f"  [+] PDF Report saved to: {pdf_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  File:       {filename}")
    print(f"  Verdict:    {verdict}")
    print(f"  Confidence: {confidence * 100:.1f}%")
    print(f"  Risk Level: {claude_result.get('risk_level', 'Unknown')}")
    print(f"  Family:     {claude_result.get('malware_family', 'None')}")
    print(f"  IOCs:       {ioc_data.get('total_iocs_count', 0)} indicators detected")
    print(f"  SHAP Chart: {shap_chart_path}")
    print(f"  PDF Report: {pdf_path}")
    print("=" * 70 + "\n")

    # Return structured result for programmatic use
    return {
        "filename": filename,
        "verdict": verdict,
        "confidence": confidence,
        "malicious_probability": malicious_prob,
        "risk_level": claude_result.get("risk_level"),
        "iocs_count": ioc_data.get("total_iocs_count", 0),
        "shap_chart": shap_chart_path,
        "pdf_report": pdf_path,
        "claude_analysis": claude_result,
        "ioc_data": ioc_data,
        "top_attributions": [(n, v, a) for n, v, a in attributions[:5]]
    }


if __name__ == "__main__":
    main()
