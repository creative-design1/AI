import numpy as np
import joblib
import tensorflow as tf
from collections import deque
import torch
import torch.nn as nn

class FallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.lstm(x)       # out: (B, T, hidden)
        last = out[:, -1, :]        # 마지막 타임스텝을 사용 (B, hidden)
        logits = self.fc(last)      # (B, num_classes)
        return logits


class Fall_Detector:
    
    def __init__(self, model_path, scaler_path=None, buffer_size=80, device='cpu'):
        self.device = torch.device(device)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        self.buffer = deque(maxlen=buffer_size)
        
        self.model = FallLSTM(input_size=132, hidden_size=128, num_layers=2).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
    
    
    def update_sequence(self, keypoints):
        if self.scaler is not None:
            keypoints = self.scaler.transform(keypoints.reshape(1, -1)).reshape(-1)
        self.buffer.append(keypoints)
        
    
    def predict(self):
        if len(self.buffer) == self.buffer.maxlen:
            seq = np.array(self.buffer, dtype=np.float32)
            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(seq)
                probs = torch.softmax(logits, dim=1)
                fall_prob = probs[0][1].item()
                
            #return fall_prob > 0.0, fall_prob
            return fall_prob > 0.75, fall_prob
        return False, 0.0
    