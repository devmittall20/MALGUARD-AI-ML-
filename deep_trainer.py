import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .deep_models import APISequenceLSTM, RawByteCNN

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class DeepModelTrainer:
    """
    Handles training of PyTorch deep learning models (LSTM for API sequences, CNN for raw bytes).
    """

    def __init__(self, data_dir: str = "data/processed", models_dir: str = "models", batch_size: int = 256, epochs: int = 2):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.batch_size = batch_size
        self.epochs = epochs
        os.makedirs(self.models_dir, exist_ok=True)
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_multimodal_data(self) -> dict:
        """
        Loads train dataset from EMBER parquet and splits it for PyTorch training.
        """
        train_path = r"C:\Users\hp\Downloads\archive\train_ember_2018_v2_features.parquet"
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"EMBER dataset not found at {train_path}")

        logger.info(f"Loading datasets for Deep Learning from {train_path}...")
        df = pd.read_parquet(train_path)

        if "Label" not in df.columns:
            raise KeyError("Dataset must contain a 'Label' column.")

        # In EMBER dataset, -1 means unlabelled. We only want 0 (benign) and 1 (malicious)
        df = df[df["Label"].isin(["0.0", "1.0", "0", "1", 0, 1])]

        y = df["Label"].astype(np.float32).values
        X = df.drop(columns=["Label"]).astype(np.float32).values

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

        logger.info(f"Parsed EMBER train: {X_train.shape}, val: {X_val.shape}")

        return {
            "train_api": X_train,
            "train_bytes": X_train,
            "train_label": y_train,
            "val_api": X_val,
            "val_bytes": X_val,
            "val_label": y_val
        }

    def train_lstm(self, data: dict) -> APISequenceLSTM:
        """Trains the APISequenceLSTM model and saves the best checkpoint."""
        logger.info("Training API Sequence LSTM model...")
        
        # Datasets and Loaders
        train_ds = TensorDataset(torch.tensor(data["train_api"]), torch.tensor(data["train_label"]))
        val_ds = TensorDataset(torch.tensor(data["val_api"]), torch.tensor(data["val_label"]))
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        
        model = APISequenceLSTM().to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float("inf")
        model_path = os.path.join(self.models_dir, "lstm_model.pth")
        
        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(batch_x).squeeze()
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    pred = model(batch_x).squeeze()
                    loss = criterion(pred, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    correct += ((pred >= 0.5) == batch_y).sum().item()
                    
            val_loss /= len(val_loader.dataset)
            val_acc = correct / len(val_loader.dataset)
            
            logger.info(f"Epoch {epoch}/{self.epochs} - Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")
            
            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved best LSTM model checkpoint with val loss: {best_val_loss:.4f}")
                
        # Load best weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def train_cnn(self, data: dict) -> RawByteCNN:
        """Trains the RawByteCNN model and saves the best checkpoint."""
        logger.info("Training Raw Byte CNN model...")
        
        # Datasets and Loaders
        train_ds = TensorDataset(torch.tensor(data["train_bytes"]), torch.tensor(data["train_label"]))
        val_ds = TensorDataset(torch.tensor(data["val_bytes"]), torch.tensor(data["val_label"]))
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        
        model = RawByteCNN().to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float("inf")
        model_path = os.path.join(self.models_dir, "cnn_model.pth")
        
        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(batch_x).squeeze()
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    pred = model(batch_x).squeeze()
                    loss = criterion(pred, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    correct += ((pred >= 0.5) == batch_y).sum().item()
                    
            val_loss /= len(val_loader.dataset)
            val_acc = correct / len(val_loader.dataset)
            
            logger.info(f"Epoch {epoch}/{self.epochs} - Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")
            
            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved best CNN model checkpoint with val loss: {best_val_loss:.4f}")
                
        # Load best weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
