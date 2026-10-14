import torch
import torch.nn as nn

class EmberFFNN(nn.Module):
    """
    Deep Feed-Forward Neural Network to learn patterns from the 2381 EMBER features.
    Replacing the old APISequenceLSTM.
    """
    def __init__(self, input_dim: int = 2381, hidden_dim: int = 256):
        super(EmberFFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class EmberCNN(nn.Module):
    """
    1D CNN classifier to extract patterns from EMBER features.
    Replacing the old RawByteCNN.
    """
    def __init__(self, input_dim: int = 2381):
        super(EmberCNN, self).__init__()
        # Reshape input to (batch_size, 1 channel, 2381 features)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Determine the shape of flat features after convolutions and pooling:
        # 2381 -> conv1 (stride 2) -> 1191 -> pool1 (kernel 2) -> 595
        # 595 -> conv2 (stride 2) -> 298 -> pool2 (kernel 2) -> 149
        # Flat feature size = 32 channels * 149 features = 4768
        self.fc1 = nn.Linear(32 * 149, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Shape: (batch_size, 1, 2381)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        return self.sigmoid(logits)

# Backwards compatibility aliases
APISequenceLSTM = EmberFFNN
RawByteCNN = EmberCNN
