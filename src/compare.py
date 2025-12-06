import numpy as np
import os, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm.notebook import tqdm
import joblib
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 재현성
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 데이터 경로 (환경에 맞게 수정)
BASE_DIR = Path.cwd().parent   # 예: src에서 실행하면 프로젝트 루트
NPZ_PATH = BASE_DIR / "data" / "processed" / "npz" / "fall_dataset.npz"

# 디바이스
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 셀 2: npz 로드 및 확인
data = np.load(NPZ_PATH)
print("keys:", data.files)   # 보통 ['X','Y']

X = data['X']   # (N, T, D)
Y = data['Y']   # (N,)

print("X dtype, shape:", X.dtype, X.shape)
print("Y dtype, shape:", Y.dtype, Y.shape)
print("positive (fall) count:", int(Y.sum()), "negative count:", len(Y)-int(Y.sum()))

# 셀 3: split (샘플 단위 stratify)
test_ratio = 0.1
val_ratio = 0.1

# stratify로 클래스 비율 유지
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, Y, test_size=test_ratio, random_state=42, stratify=Y
)

val_relative = val_ratio / (1 - test_ratio)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=val_relative, random_state=42, stratify=y_trainval
)

print("train:", X_train.shape, y_train.shape)
print("val:  ", X_val.shape, y_val.shape)
print("test: ", X_test.shape, y_test.shape)

# 셀 4: scaler fit on train only -> transform all
Ntr, T, D = X_train.shape
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, D))  # (Ntr*T, D)

def apply_scaler(X_arr, scaler):
    N, T, D = X_arr.shape
    X2 = X_arr.reshape(-1, D)
    X2 = scaler.transform(X2)
    return X2.reshape(N, T, D)

X_train = apply_scaler(X_train, scaler)
X_val   = apply_scaler(X_val, scaler)
X_test  = apply_scaler(X_test, scaler)

# scaler 저장 (추론 시 필요)

joblib.dump(scaler, BASE_DIR / "src" / "scaler.save")
print("Scaler saved.")


# 셀 5: Dataset & DataLoader
class FallDataset(Dataset):
    def __init__(self, X_np, y_np):
        self.X = X_np.astype(np.float32)  # (N,T,D)
        self.y = y_np.astype(np.int64)    # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]      # (T, D)
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

batch_size = 32
train_loader = DataLoader(FallDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(FallDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(FallDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

"""
# 셀 6: LSTM 모델
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
        out, (h_n, c_n) = self.lstm(x)  # out: (B, T, hidden_size)
        last = out[:, -1, :]            # 마지막 타임스텝 feature (B, hidden)
        logits = self.fc(last)          # (B, num_classes)
        return logits

input_size = X_train.shape[2]
model = FallLSTM(input_size=input_size).to(device)
print(model)
"""

"""
class FallGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        # LSTM 대신 GRU 사용
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, D)
        # GRU는 out, h_n 만 반환
        out, h_n = self.gru(x) 
        # h_n[-1]은 마지막 레이어의 마지막 시점 hidden state
        last = h_n[-1] 
        # out[:, -1, :] 대신 h_n[-1]을 사용하는 것이 일반적입니다. (bidirectional이 아닐 때)
        
        logits = self.fc(last)
        return logits

input_size = X_train.shape[2]
# 모델 초기화
model = FallGRU(input_size=input_size).to(device)
print(model)
"""

"""
# 셀 6: 1D-CNN 모델
import torch.nn.functional as F

class CNN1DModel(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super().__init__()
        
        # Conv1d는 (B, D, T) 형태의 입력을 받으므로, D (특징 차원)이 in_channels가 됩니다.
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        
        # 시퀀스 길이에 관계없이 고정된 크기(1)로 평균 풀링하여 시퀀스 정보를 압축
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1) 
        
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, D) -> Conv1D를 위해 permute: (B, D, T)
        x = x.permute(0, 2, 1) 
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # (B, 128, T') -> (B, 128, 1) -> (B, 128)
        x = self.adaptive_pool(x).squeeze(-1) 
        
        out = self.classifier(x)
        return out

input_size = X_train.shape[2]
# 모델 초기화
model = CNN1DModel(input_size=input_size).to(device)
print(model)
"""

"""
# 셀 6: Transformer Encoder 모델
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 트랜스포머 하이퍼파라미터 설정
HIDDEN_SIZE = 128  # d_model
NUM_LAYERS = 2
NHEAD = 4          # Multi-Head Attention의 헤드 수 (d_model의 약수가 좋음)
DROPOUT = 0.3      # 기존 RNN과 동일하게 설정

class TransformerModel(nn.Module):
    def __init__(self, input_size, nhead, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        
        d_model = hidden_size
        
        # 입력 특징 차원(13)을 모델 차원(d_model)로 매핑
        self.input_linear = nn.Linear(input_size, d_model)
        
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # 시퀀스 전체 정보를 압축하기 위해 평균 풀링 사용
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, D)
        
        # 1. 입력 특징 매핑: (B, T, D) -> (B, T, d_model)
        x = self.input_linear(x)
        
        # 2. Transformer Encoder 통과 (Positional Encoding은 생략)
        output = self.transformer_encoder(x) # (B, T, d_model)
        
        # 3. 글로벌 풀링
        x = output.permute(0, 2, 1) # (B, d_model, T)로 변환
        x = self.global_pool(x).squeeze(-1) # (B, d_model)
        
        out = self.classifier(x)
        return out

input_size = X_train.shape[2]
# 모델 초기화
model = TransformerModel(
    input_size, nhead=NHEAD, hidden_size=HIDDEN_SIZE, 
    num_layers=NUM_LAYERS, num_classes=2, dropout=DROPOUT
).to(device)
print(model)
"""

# 셀 7: loss / optimizer / scheduler
# 클래스 불균형 보정
unique, counts = np.unique(y_train, return_counts=True)
print("train class counts:", dict(zip(unique, counts)))
n = len(y_train)
w0 = n / (2 * counts[0]) if counts[0] > 0 else 1.0
w1 = n / (2 * counts[1]) if len(counts)>1 and counts[1]>0 else 1.0
weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
print("class weights:", weights.cpu().numpy())

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# 셀 8: train/eval helpers

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    preds_all, labels_all = [], []
    for Xb, yb in tqdm(loader, desc="Train", leave=False):
        Xb = Xb.to(device); yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits.detach(), dim=1).cpu().numpy()
        preds_all.extend(preds); labels_all.extend(yb.cpu().numpy())

    avg_loss = float(np.mean(losses))
    acc = accuracy_score(labels_all, preds_all)
    prec, rec, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average='binary', zero_division=0)
    return avg_loss, acc, prec, rec, f1

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all, labels_all = [], []
    with torch.no_grad():
        for Xb, yb in tqdm(loader, desc="Eval", leave=False):
            Xb = Xb.to(device); yb = yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_all.extend(preds); labels_all.extend(yb.cpu().numpy())

    avg_loss = float(np.mean(losses))
    acc = accuracy_score(labels_all, preds_all)
    prec, rec, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average='binary', zero_division=0)
    cm = confusion_matrix(labels_all, preds_all)
    return avg_loss, acc, prec, rec, f1, cm

# 셀 9: training loop
num_epochs = 30
best_val_f1 = 0.0
ckpt_dir = BASE_DIR / "models"
ckpt_dir.mkdir(parents=True, exist_ok=True)
best_ckpt = ckpt_dir / "best_fall_model_test.pth"

history = {"train_loss":[], "val_loss":[], "train_f1":[], "val_f1":[]}

for epoch in range(1, num_epochs+1):
    print(f"\n=== Epoch {epoch}/{num_epochs} ===")
    tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train: loss={tr_loss:.4f} acc={tr_acc:.3f} prec={tr_prec:.3f} rec={tr_rec:.3f} f1={tr_f1:.3f}")

    val_loss, val_acc, val_prec, val_rec, val_f1, cm = eval_one_epoch(model, val_loader, criterion, device)
    print(f"Val:   loss={val_loss:.4f} acc={val_acc:.3f} prec={val_prec:.3f} rec={val_rec:.3f} f1={val_f1:.3f}")
    print("Confusion matrix:\n", cm)

    history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
    history["train_f1"].append(tr_f1); history["val_f1"].append(val_f1)

    scheduler.step(val_f1)

    # best 저장
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_f1": val_f1
        }, str(best_ckpt))
        print("Saved best checkpoint:", best_ckpt)

print("Training finished. Best val F1:", best_val_f1)
# history 저장
joblib.dump(history, str(ckpt_dir / "history.joblib"))

# 셀 10: load best and test
ck = torch.load(str(best_ckpt), map_location=device)
model.load_state_dict(ck["model_state"])
model.to(device)
model.eval()

test_loss, test_acc, test_prec, test_rec, test_f1, test_cm = eval_one_epoch(model, test_loader, criterion, device)
print("\n=== TEST RESULTS ===")
print(f"Loss: {test_loss:.4f} Acc: {test_acc:.3f} Prec: {test_prec:.3f} Rec: {test_rec:.3f} F1: {test_f1:.3f}")
print("Confusion matrix:\n", test_cm)