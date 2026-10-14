import os
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """
    Computes SHAP values using TreeExplainer on trained classifiers to explain predictions,
    and renders feature attribution bar charts.
    """

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
        # Initialize explainer defensively
        try:
            import shap
            # XGBoost or RandomForest models are supported by TreeExplainer
            # If the model is a pipeline (Random Forest), explain the classifier step
            from sklearn.pipeline import Pipeline
            if isinstance(self.model, Pipeline):
                base_model = self.model.named_steps["classifier"]
            else:
                base_model = self.model

            self.explainer = shap.TreeExplainer(base_model)
            logger.info("SHAP TreeExplainer initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP TreeExplainer (falling back to heuristic explanations): {e}")

    def explain_sample(self, X_sample: pd.DataFrame) -> List[Tuple[str, float, float]]:
        """
        Calculates feature attributions (SHAP values) for a single sample.
        Returns a sorted list of Tuples (feature_name, shap_value, actual_value) for top 10 features.
        """
        # Ensure input is a DataFrame with correct columns
        if isinstance(X_sample, np.ndarray):
            X_sample = pd.DataFrame(X_sample, columns=self.feature_names)
            
        sample_row = X_sample.iloc[0]
        attributions = []

        if self.explainer is not None:
            try:
                # If pipeline (Random Forest), we must scale the features first
                from sklearn.pipeline import Pipeline
                if isinstance(self.model, Pipeline):
                    scaler = self.model.named_steps["scaler"]
                    X_input = scaler.transform(X_sample)
                else:
                    X_input = X_sample.values

                # Compute SHAP values
                shap_values = self.explainer.shap_values(X_input)
                
                # shap_values can be a list (for multi-class/binary RF) or a single array (for binary XGBoost)
                if isinstance(shap_values, list):
                    # For binary classification, shap_values[1] is the attribution for class 1 (Malicious)
                    shap_vals = shap_values[1][0]
                elif shap_values.ndim == 3:
                    # In some versions it returns (n_samples, n_features, n_classes)
                    shap_vals = shap_values[0, :, 1]
                elif shap_values.ndim == 2:
                    # For binary XGBoost, it might return a 2D array (n_samples, n_features) representing class 1
                    shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values

                for name, val in zip(self.feature_names, shap_vals):
                    attributions.append((name, float(val), float(sample_row[name])))
            except Exception as e:
                logger.error(f"Error computing SHAP values: {e}. Falling back to heuristic.")
                self.explainer = None # Fall through to heuristic

        # Heuristic/Rule-based Attribution Fallback (runs if shap is missing or errors)
        if self.explainer is None or not attributions:
            logger.info("Using heuristic/rule-based feature attribution.")
            # Map features that typically distinguish malicious vs benign based on our distribution rules
            # We compare the sample values to average benign/malicious profiles
            for name in self.feature_names:
                val = sample_row[name]
                contrib = 0.0
                
                # Heuristics based on static feature distributions
                if name == "file_entropy" and val > 6.8:
                    contrib = 0.3 * (val - 6.8)
                elif name == "sec_entropy_max" and val > 7.2:
                    contrib = 0.35 * (val - 7.2)
                elif name == "imports_dll_count" and val < 3:
                    contrib = 0.15 * (3 - val)
                elif name.startswith("import_") and val == 1:
                    # Suspicious APIs like VirtualAlloc
                    suspicious = ["VirtualAlloc", "WriteProcessMemory", "CreateRemoteThread", "GetProcAddress"]
                    if any(s in name for s in suspicious):
                        contrib = 0.25
                elif name.startswith("yara_") and val == 1:
                    contrib = 0.4
                elif name == "opcode_freq_xor" and val > 0.06:
                    contrib = 0.25 * (val - 0.06)
                elif name == "opcode_freq_mov" and val < 0.28:
                    contrib = 0.15 * (0.28 - val)
                elif name == "sec_writable_count" and val > 2:
                    contrib = 0.1 * (val - 2)
                    
                attributions.append((name, contrib, float(val)))

        # Sort by absolute SHAP value / contribution descending
        attributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return attributions[:10]

    def plot_attributions(self, attributions: List[Tuple[str, float, float]], out_dir: str = "reports") -> str:
        """
        Renders and saves a horizontal bar chart of local feature attributions.
        Positive attributions (red) pull prediction towards malicious; negative (blue) towards benign.
        """
        os.makedirs(out_dir, exist_ok=True)
        
        # Unzip values
        names, values, actuals = zip(*reversed(attributions))
        
        # Clean names for plotting
        display_names = []
        for name, act in zip(names, actuals):
            clean_name = name.replace("opt_", "").replace("header_", "").replace("sec_", "").replace("imports_", "")
            display_names.append(f"{clean_name} = {act:.2f}")

        # Plot
        plt.figure(figsize=(9, 5), dpi=150)
        colors = ["#ff4d4d" if v >= 0 else "#4da6ff" for v in values] # Red for malicious-driving, Blue for benign-driving
        
        bars = plt.barh(display_names, values, color=colors, height=0.6, edgecolor="grey", alpha=0.95)
        
        # Add labels
        plt.axvline(x=0, color="black", lw=1.0, linestyle="--")
        plt.xlabel("Feature Contribution (Impact on Threat Score)", fontsize=10)
        plt.title("Explainable AI (XAI) Local Feature Attributions", fontsize=12, fontweight="bold", pad=15)
        plt.grid(True, axis="x", linestyle=":", alpha=0.6)
        
        # Style layout
        for bar in bars:
            width = bar.get_width()
            # Annotate contribution score next to bar
            if width >= 0:
                plt.text(width + 0.01, bar.get_y() + 0.15, f"+{width:.3f}", va="center", ha="left", fontsize=8, color="red", fontweight="bold")
            else:
                plt.text(width - 0.01, bar.get_y() + 0.15, f"{width:.3f}", va="center", ha="right", fontsize=8, color="blue", fontweight="bold")

        plt.tight_layout()
        out_path = os.path.join(out_dir, "shap_importance.png")
        plt.savefig(out_path)
        plt.close()
        logger.info(f"SHAP importance chart saved to {out_path}")
        return out_path
