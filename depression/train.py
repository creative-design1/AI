import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# -----------------------------------
# 1) 데이터 불러오기
# -----------------------------------
X = np.load("emb_ko_sbert.npy")
y = np.loadtxt("phq_labels.txt")  # 정수형 PHQ 점수

print("임베딩 shape:", X.shape)
print("라벨 shape:", y.shape)

# -----------------------------------
# 2) Train / Test Split (80% / 20%)
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# -----------------------------------
# 3) XGBoost 회귀 모델 생성
# -----------------------------------
model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",  # 빠른 학습
    random_state=42
)

# -----------------------------------
# 4) 모델 학습
# -----------------------------------
print("학습 시작...")
model.fit(X_train, y_train)
print("학습 완료!")

# -----------------------------------
# 5) 테스트 평가
# -----------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n### 평가 결과 ###")
print("MAE (절대 오차):", round(mae, 3))
print("RMSE:", round(rmse, 3))

# -----------------------------------
# 6) 모델 저장
# -----------------------------------
joblib.dump(model, "xgboost_ko_sbert.pkl")
print("\n모델 저장 완료: xgboost_ko_sbert.pkl")

# -----------------------------------
# 7) 예측 샘플 출력
# -----------------------------------
print("\n### 예측 샘플 ###")
for i in range(5):
    print(f"실제 PHQ: {y_test[i]}, 예측 PHQ: {y_pred[i]:.2f}")
