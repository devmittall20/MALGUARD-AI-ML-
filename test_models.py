import os
import json
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
import joblib
from src.data.dataset import DatasetManager
from src.models.classical_trainer import ModelTrainer
from src.models.ensemble import PEVotingEnsemble
from src.models.evaluator import ModelEvaluator

@pytest.fixture
def temp_project_dir():
    # Setup temporary directory structure for tests
    d = tempfile.mkdtemp()
    data_dir = os.path.join(d, "processed")
    models_dir = os.path.join(d, "models")
    reports_dir = os.path.join(d, "reports")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate mock dataset files
    manager = DatasetManager(data_root=d)
    
    # Train set (100 samples)
    df_train = manager.generate_synthetic_dataset(n_samples=100, benign_ratio=0.5)
    df_train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    
    # Val set (40 samples)
    df_val = manager.generate_synthetic_dataset(n_samples=40, benign_ratio=0.5)
    df_val.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    
    # Test set (40 samples)
    df_test = manager.generate_synthetic_dataset(n_samples=40, benign_ratio=0.5)
    df_test.to_csv(os.path.join(data_dir, "test.csv"), index=False)

    yield d, data_dir, models_dir, reports_dir

    # Teardown
    if os.path.exists(d):
        shutil.rmtree(d)

def test_model_training_and_loading(temp_project_dir):
    _, data_dir, models_dir, _ = temp_project_dir
    
    # 1. Initialize trainer
    trainer = ModelTrainer(data_dir=data_dir, models_dir=models_dir)
    
    # 2. Load data
    X_train, y_train, X_val, y_val = trainer.load_data()
    assert X_train.shape == (100, 105) # 106 cols minus label
    assert X_val.shape == (40, 105)
    assert y_train.shape == (100,)
    assert y_val.shape == (40,)
    
    # 3. Train models
    rf_pipeline = trainer.train_random_forest(X_train, y_train)
    xgb_model = trainer.train_xgboost(X_train, y_train, X_val, y_val)
    gb_model = trainer.train_gradient_boosting(X_train, y_train)
    
    # Check model pickle files were saved
    assert os.path.exists(os.path.join(models_dir, "rf_model.pkl"))
    assert os.path.exists(os.path.join(models_dir, "xgb_model.pkl"))
    assert os.path.exists(os.path.join(models_dir, "gb_model.pkl"))

    # Check loaded model types
    loaded_rf = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
    loaded_xgb = joblib.load(os.path.join(models_dir, "xgb_model.pkl"))
    loaded_gb = joblib.load(os.path.join(models_dir, "gb_model.pkl"))
    
    # Run simple prediction to check model load sanity
    assert len(loaded_rf.predict(X_val)) == 40
    assert len(loaded_xgb.predict(X_val)) == 40
    assert len(loaded_gb.predict(X_val)) == 40

def test_soft_voting_ensemble(temp_project_dir):
    _, data_dir, models_dir, _ = temp_project_dir
    
    # Train submodels
    trainer = ModelTrainer(data_dir=data_dir, models_dir=models_dir)
    X_train, y_train, X_val, y_val = trainer.load_data()
    rf = trainer.train_random_forest(X_train, y_train)
    xgb = trainer.train_xgboost(X_train, y_train, X_val, y_val)
    gb = trainer.train_gradient_boosting(X_train, y_train)
    
    # Build ensemble
    ensemble = PEVotingEnsemble(rf_model=rf, xgb_model=xgb, gb_model=gb)
    
    # Predict probabilities
    probs = ensemble.predict_proba(X_val)
    assert probs.shape == (40, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    
    # Verify soft voting math: with no LSTM/CNN loaded, fallback duplicates RF/XGBoost
    # Ensemble averages: (rf + xgb + gb + lstm_fallback=xgb + cnn_fallback=rf) / 5
    rf_p = rf.predict_proba(X_val)
    xgb_p = xgb.predict_proba(X_val)
    gb_p = gb.predict_proba(X_val)
    expected_avg = (rf_p + xgb_p + gb_p + xgb_p + rf_p) / 5.0
    assert np.allclose(probs, expected_avg, atol=1e-6)

    # Predict class labels
    preds = ensemble.predict(X_val, threshold=0.5)
    assert preds.shape == (40,)
    assert set(np.unique(preds)).issubset({0, 1})
    
    # Test saving/loading ensemble
    ensemble_path = os.path.join(models_dir, "ensemble_model.pkl")
    ensemble.save(ensemble_path)
    assert os.path.exists(ensemble_path)
    
    loaded_ensemble = PEVotingEnsemble.load(ensemble_path)
    loaded_ensemble.load_submodels(models_dir)
    
    loaded_probs = loaded_ensemble.predict_proba(X_val)
    assert np.allclose(probs, loaded_probs)

def test_evaluator_reporting(temp_project_dir):
    _, data_dir, models_dir, reports_dir = temp_project_dir
    
    # Train and build ensemble
    trainer = ModelTrainer(data_dir=data_dir, models_dir=models_dir)
    X_train, y_train, X_val, y_val = trainer.load_data()
    rf = trainer.train_random_forest(X_train, y_train)
    xgb = trainer.train_xgboost(X_train, y_train, X_val, y_val)
    gb = trainer.train_gradient_boosting(X_train, y_train)
    ensemble = PEVotingEnsemble(rf_model=rf, xgb_model=xgb, gb_model=gb)
    
    # Evaluate
    evaluator = ModelEvaluator(data_dir=data_dir, reports_dir=reports_dir)
    X_test, y_test = evaluator.load_test_data()
    
    results = {}
    results["RF"] = evaluator.evaluate_model(rf, "RF", X_test, y_test)
    results["Ensemble"] = evaluator.evaluate_model(ensemble, "Ensemble", X_test, y_test)
    
    # Save reports
    evaluator.save_metrics_to_json(results, os.path.join(reports_dir, "model_metrics.json"))
    roc_path = evaluator.plot_roc_curves(results, y_test)
    cm_path = evaluator.plot_confusion_matrices(results, y_test)
    
    assert os.path.exists(os.path.join(reports_dir, "model_metrics.json"))
    assert os.path.exists(roc_path)
    assert os.path.exists(cm_path)
    
    # Check structure of JSON
    with open(os.path.join(reports_dir, "model_metrics.json"), "r") as f:
        metrics = json.load(f)
    assert "RF" in metrics
    assert "Ensemble" in metrics
    assert "accuracy" in metrics["RF"]
    assert "roc_auc" in metrics["Ensemble"]
