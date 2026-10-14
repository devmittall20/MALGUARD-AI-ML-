import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import torch
from src.models.deep_models import APISequenceLSTM, RawByteCNN
from src.models.deep_trainer import DeepModelTrainer
from src.models.classical_trainer import ModelTrainer
from src.models.ensemble import PEVotingEnsemble
from src.models.evaluator import ModelEvaluator

def test_deep_models_forward():
    # 1. Test LSTM
    lstm = APISequenceLSTM(vocab_size=32, embedding_dim=16, hidden_dim=32)
    # Batch size = 4, Sequence length = 50
    input_api = torch.randint(0, 32, (4, 50))
    out_api = lstm(input_api)
    assert out_api.shape == (4, 1)
    assert torch.all(out_api >= 0.0)
    assert torch.all(out_api <= 1.0)
    
    # 2. Test CNN
    cnn = RawByteCNN(input_dim=1024)
    # Batch size = 4, Byte length = 1024
    input_bytes = torch.rand(4, 1024)
    out_bytes = cnn(input_bytes)
    assert out_bytes.shape == (4, 1)
    assert torch.all(out_bytes >= 0.0)
    assert torch.all(out_bytes <= 1.0)

def test_voting_ensemble_multimodal():
    # Set up mock classical classifiers
    class DummyClassifier:
        def predict_proba(self, X):
            # Return uniform predictions of shape (N, 2)
            n_samples = len(X)
            probs = np.zeros((n_samples, 2))
            probs[:, 0] = 0.6
            probs[:, 1] = 0.4
            return probs
            
        def predict(self, X):
            return np.zeros(len(X), dtype=np.int32)

    rf_mock = DummyClassifier()
    xgb_mock = DummyClassifier()
    
    # Set up mock PyTorch classifiers
    lstm_mock = APISequenceLSTM()
    cnn_mock = RawByteCNN()
    
    # Force models to output deterministic values for testing
    ensemble = PEVotingEnsemble(
        rf_model=rf_mock,
        xgb_model=xgb_mock,
        lstm_model=lstm_mock,
        cnn_model=cnn_mock
    )
    
    # Create a small DataFrame representing test samples (2 samples)
    data = {}
    # Static features (just some random names to match schema)
    for col in ["file_size", "file_entropy", "header_machine"]:
        data[col] = [1000.0, 5.0, 332.0]
    # API seq features
    for i in range(50):
        data[f"api_seq_{i}"] = [float(i % 32)] * 3
    # Byte features
    for i in range(1024):
        data[f"byte_{i}"] = [0.5] * 3
        
    df_test = pd.DataFrame(data)
    
    # Evaluate predict_proba
    probs = ensemble.predict_proba(df_test)
    assert probs.shape == (3, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    
    # Evaluate predict
    preds = ensemble.predict(df_test)
    assert preds.shape == (3,)
    assert np.all((preds == 0) | (preds == 1))
