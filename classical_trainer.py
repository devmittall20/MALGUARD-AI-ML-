import os
import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Configure logging if not already configured
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class ModelTrainer:
    """
    Handles loading of data and training of Random Forest, XGBoost, and Gradient Boosting models.
    """

    def __init__(self, data_dir: str = "data/processed", models_dir: str = "models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Loads train dataset from EMBER parquet and splits into train/val.
        """
        train_path = r"C:\Users\hp\Downloads\archive\train_ember_2018_v2_features.parquet"
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"EMBER dataset not found at {train_path}")

        logger.info(f"Loading EMBER dataset from {train_path}...")
        df = pd.read_parquet(train_path)

        if "Label" not in df.columns:
            raise KeyError("Dataset must contain a 'Label' column.")

        # In EMBER dataset, -1 means unlabelled. We only want 0 (benign) and 1 (malicious)
        df = df[df["Label"].isin(["0.0", "1.0", "0", "1", 0, 1])]
        
        y = df["Label"].astype(np.float32)
        X = df.drop(columns=["Label"]).astype(np.float32)

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

        logger.info(f"Loaded Train: {X_train.shape}, Val: {X_val.shape}")
        return X_train, y_train, X_val, y_val

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Trains a Random Forest classifier wrapped in a Pipeline.
        """
        logger.info("Training Random Forest classifier...")
        base_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", base_rf)
        ])
        
        pipeline.fit(X_train, y_train)
        
        best_pipeline = pipeline
        
        model_path = os.path.join(self.models_dir, "rf_model.pkl")
        joblib.dump(best_pipeline, model_path)
        logger.info(f"Random Forest model saved to {model_path}")
        return best_pipeline

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> XGBClassifier:
        """
        Trains an XGBoost classifier.
        """
        logger.info("Training XGBoost classifier...")
        
        best_xgb = XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
        
        best_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        model_path = os.path.join(self.models_dir, "xgb_model.pkl")
        joblib.dump(best_xgb, model_path)
        logger.info(f"XGBoost model saved to {model_path}")
        return best_xgb

    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
        """
        Trains a Gradient Boosting classifier.
        """
        logger.info("Training Gradient Boosting classifier...")
        gb_clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
        
        gb_clf.fit(X_train, y_train)
        
        model_path = os.path.join(self.models_dir, "gb_model.pkl")
        joblib.dump(gb_clf, model_path)
        logger.info(f"Gradient Boosting model saved to {model_path}")
        return gb_clf
