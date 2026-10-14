import os
import joblib
import numpy as np
import torch
from typing import Any, Optional
from .deep_models import APISequenceLSTM, RawByteCNN

class PEVotingEnsemble:
    """
    Custom Soft Voting Ensemble Classifier that aggregates predictions from 
    Random Forest, XGBoost, LSTM (API call sequence), and CNN (raw bytes) models.
    """

    def __init__(
        self, 
        rf_model: Optional[Any] = None, 
        xgb_model: Optional[Any] = None, 
        gb_model: Optional[Any] = None,
        lstm_model: Optional[Any] = None,
        cnn_model: Optional[Any] = None
    ):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.gb_model = gb_model # Kept for backward compatibility
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model

    def load_submodels(self, models_dir: str = "models") -> None:
        """
        Loads the individual sub-models from the specified directory.
        """
        rf_path = os.path.join(models_dir, "rf_model.pkl")
        xgb_path = os.path.join(models_dir, "xgb_model.pkl")
        gb_path = os.path.join(models_dir, "gb_model.pkl")
        lstm_path = os.path.join(models_dir, "lstm_model.pth")
        cnn_path = os.path.join(models_dir, "cnn_model.pth")

        if not os.path.exists(rf_path) or not os.path.exists(xgb_path):
            raise FileNotFoundError(f"One or more sub-models RF/XGBoost are missing from {models_dir}. Run trainer first.")

        self.rf_model = joblib.load(rf_path)
        self.xgb_model = joblib.load(xgb_path)
        
        # Load Gradient Boosting if it exists (for backward compatibility)
        if os.path.exists(gb_path):
            self.gb_model = joblib.load(gb_path)
        else:
            self.gb_model = None

        # Load LSTM API call sequence model
        if os.path.exists(lstm_path):
            self.lstm_model = APISequenceLSTM()
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
        else:
            self.lstm_model = None

        # Load CNN raw byte model
        if os.path.exists(cnn_path):
            self.cnn_model = RawByteCNN()
            self.cnn_model.load_state_dict(torch.load(cnn_path, map_location="cpu"))
        else:
            self.cnn_model = None

    def save(self, path: str = "models/ensemble_model.pkl") -> None:
        """
        Saves the voting ensemble reference/state.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Temporarily clear torch model attributes before pickling to avoid serialization issues
        temp_lstm = self.lstm_model
        temp_cnn = self.cnn_model
        self.lstm_model = None
        self.cnn_model = None
        
        joblib.dump(self, path)
        
        # Restore them
        self.lstm_model = temp_lstm
        self.cnn_model = temp_cnn

    @classmethod
    def load(cls, path: str = "models/ensemble_model.pkl") -> "PEVotingEnsemble":
        """
        Loads the saved voting ensemble model.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ensemble file not found: {path}")
        return joblib.load(path)

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Calculates the average probability prediction of all four ensemble classifiers.
        """
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("RF and XGBoost sub-models must be loaded/defined before predicting.")

        # Identify features
        api_cols = [f"api_seq_{i}" for i in range(50)]
        byte_cols = [f"byte_{i}" for i in range(1024)]

        # 1. Classical Static Classifiers
        cols_to_drop = [c for c in X.columns if c.startswith("api_seq_") or c.startswith("byte_")]
        X_static = X.drop(columns=cols_to_drop)

        rf_probs = self.rf_model.predict_proba(X_static)
        xgb_probs = self.xgb_model.predict_proba(X_static)

        # 2. PyTorch Classifiers (LSTM & CNN)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prepare probability placeholders
        lstm_probs = np.zeros_like(rf_probs)
        cnn_probs = np.zeros_like(rf_probs)
        
        # LSTM API Sequence
        if self.lstm_model is not None:
            self.lstm_model.to(device)
            self.lstm_model.eval()
            if "api_seq_0" in X.columns:
                X_api = X[api_cols].values.astype(np.float32)
            else:
                X_api = X.values.astype(np.float32)
            
            lstm_preds = []
            batch_size = 1024
            with torch.no_grad():
                for i in range(0, len(X_api), batch_size):
                    batch_x = X_api[i:i+batch_size]
                    tensor_x = torch.tensor(batch_x).to(device)
                    pred = self.lstm_model(tensor_x).cpu().numpy()
                    pred = pred.squeeze(axis=-1) if len(pred.shape) > 1 else pred
                    if pred.ndim == 0:
                        pred = np.array([pred])
                    lstm_preds.append(pred)
            lstm_concat = np.concatenate(lstm_preds)
            lstm_probs[:, 1] = lstm_concat
            lstm_probs[:, 0] = 1.0 - lstm_concat
        else:
            # Fallback to XGBoost if PyTorch submodel is not loaded
            lstm_probs = xgb_probs

        # CNN Raw Bytes
        if self.cnn_model is not None:
            self.cnn_model.to(device)
            self.cnn_model.eval()
            if "byte_0" in X.columns:
                X_bytes = X[byte_cols].values.astype(np.float32)
            else:
                X_bytes = X.values.astype(np.float32)
            
            cnn_preds = []
            batch_size = 1024
            with torch.no_grad():
                for i in range(0, len(X_bytes), batch_size):
                    batch_x = X_bytes[i:i+batch_size]
                    tensor_x = torch.tensor(batch_x).to(device)
                    pred = self.cnn_model(tensor_x).cpu().numpy()
                    pred = pred.squeeze(axis=-1) if len(pred.shape) > 1 else pred
                    if pred.ndim == 0:
                        pred = np.array([pred])
                    cnn_preds.append(pred)
            cnn_concat = np.concatenate(cnn_preds)
            cnn_probs[:, 1] = cnn_concat
            cnn_probs[:, 0] = 1.0 - cnn_concat
        else:
            # Fallback to RF if PyTorch submodel is not loaded
            cnn_probs = rf_probs

        # Evaluate ensemble average
        if self.gb_model is not None:
            # Support 5-model ensemble if Gradient Boosting was trained/restored
            gb_probs = self.gb_model.predict_proba(X_static)
            avg_probs = (rf_probs + xgb_probs + gb_probs + lstm_probs + cnn_probs) / 5.0
        else:
            # Standard 4-model ensemble
            avg_probs = (rf_probs + xgb_probs + lstm_probs + cnn_probs) / 4.0

        return avg_probs

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """
        Predicts binary class label (0 for benign, 1 for malicious) based on 
        averaged probabilities and a classification threshold.
        """
        probs = self.predict_proba(X)
        class1_probs = probs[:, 1]
        preds = (class1_probs >= threshold).astype(np.int32)
        return preds
