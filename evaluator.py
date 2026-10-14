import os
import json
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/command-line plots
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)

logger = logging.getLogger(__name__)

# Configure logging if not already configured
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class ModelEvaluator:
    """
    Evaluates ML models on test datasets and generates JSON metrics reports 
    and comparison charts (ROC curves, Confusion Matrices).
    """

    def __init__(self, data_dir: str = "data/processed", reports_dir: str = "reports"):
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Loads the test EMBER parquet dataset.
        """
        test_path = r"C:\Users\hp\Downloads\archive\test_ember_2018_v2_features.parquet"
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test parquet not found at {test_path}")

        logger.info(f"Loading test dataset from {test_path}...")
        df = pd.read_parquet(test_path)

        if "Label" not in df.columns:
            raise KeyError("Test dataset must contain a 'Label' column.")

        # Filter out unlabelled
        df = df[df["Label"].isin(["0.0", "1.0", "0", "1", 0, 1])]

        y_test = df["Label"].astype(np.float32)
        X_test = df.drop(columns=["Label"]).astype(np.float32)

        logger.info(f"Loaded Test features shape: {X_test.shape}")
        return X_test, y_test

    def evaluate_model(self, model: Any, name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Runs evaluation on a single model or ensemble, returning predictions, 
        probabilities, and metric scores.
        """
        logger.info(f"Evaluating model: {name}...")
        
        # Drop sequence/byte columns for individual classical classifiers
        if "Ensemble" not in name:
            cols_to_drop = [c for c in X_test.columns if c.startswith("api_seq_") or c.startswith("byte_")]
            X_eval = X_test.drop(columns=cols_to_drop)
        else:
            X_eval = X_test

        # Predict labels and probabilities
        if hasattr(model, "predict") and hasattr(model, "predict_proba"):
            # Predict labels and probabilities for sklearn/xgboost
            preds = model.predict(X_eval)
            probs = model.predict_proba(X_eval)
            
            # If classifier outputs binary probabilities, take the positive class (malicious)
            if probs.ndim == 2:
                class1_probs = probs[:, 1]
            else:
                class1_probs = probs
        else:
            # PyTorch model evaluation
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            
            # Determine features based on class name
            class_name = model.__class__.__name__
            if "LSTM" in class_name or "EmberFFNN" in class_name:
                api_cols = [f"api_seq_{i}" for i in range(50)]
                if "api_seq_0" in X_test.columns:
                    X_arr = X_test[api_cols].values.astype(np.float32)
                else:
                    X_arr = X_test.values.astype(np.float32)
            else:
                byte_cols = [f"byte_{i}" for i in range(1024)]
                if "byte_0" in X_test.columns:
                    X_arr = X_test[byte_cols].values.astype(np.float32)
                else:
                    X_arr = X_test.values.astype(np.float32)
                
            preds_list = []
            batch_size = 1024
            with torch.no_grad():
                for i in range(0, len(X_arr), batch_size):
                    batch_x = X_arr[i:i+batch_size]
                    tensor_x = torch.tensor(batch_x).to(device)
                    prob_pred = model(tensor_x).cpu().numpy()
                    prob_pred = prob_pred.squeeze(axis=-1) if len(prob_pred.shape) > 1 else prob_pred
                    if prob_pred.ndim == 0:
                        prob_pred = np.array([prob_pred])
                    preds_list.append(prob_pred)
            
            class1_probs = np.concatenate(preds_list)
            preds = (class1_probs >= 0.5).astype(np.int32)

        # Compute metric scores
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        auc = float(roc_auc_score(y_test, class1_probs))

        logger.info(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "preds": preds,
            "probs": class1_probs
        }

    def plot_roc_curves(self, results: Dict[str, Dict[str, Any]], y_test: pd.Series) -> str:
        """
        Plots the ROC curves of all evaluated models together and saves the figure.
        """
        plt.figure(figsize=(8, 6), dpi=150)
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Guess (AUC = 0.50)")
        
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(y_test, res["probs"])
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {res['roc_auc']:.4f})")
            
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate", fontsize=11)
        plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        
        out_path = os.path.join(self.reports_dir, "roc_curve.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        logger.info(f"ROC curves plot saved to {out_path}")
        return out_path

    def plot_confusion_matrices(self, results: Dict[str, Dict[str, Any]], y_test: pd.Series) -> str:
        """
        Plots confusion matrices of all models in a 2x2 grid and saves the figure.
        """
        models_count = len(results)
        if models_count == 0:
            return ""

        # Set up a grid depending on model counts (up to 4 models)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
        axes = axes.flatten()

        for idx, (name, res) in enumerate(results.items()):
            if idx >= 4:
                break
            
            cm = confusion_matrix(y_test, res["preds"])
            ax = axes[idx]
            
            # Draw confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malicious"])
            disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
            ax.set_title(f"{name} Confusion Matrix", fontsize=11, fontweight="bold")
            
        # Hide any unused axes
        for j in range(models_count, 4):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        out_path = os.path.join(self.reports_dir, "confusion_matrix.png")
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Confusion matrices plot saved to {out_path}")
        return out_path

    def save_metrics_to_json(self, results: Dict[str, Dict[str, Any]], path: str = "reports/model_metrics.json") -> None:
        """
        Saves model performance metrics (excluding predictions and probabilities) to a JSON file.
        """
        export_metrics = {}
        for name, res in results.items():
            export_metrics[name] = {
                "accuracy": res["accuracy"],
                "precision": res["precision"],
                "recall": res["recall"],
                "f1_score": res["f1_score"],
                "roc_auc": res["roc_auc"]
            }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(export_metrics, f, indent=4)
        logger.info(f"Model metrics saved to {path}")
