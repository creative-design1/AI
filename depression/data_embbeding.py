from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# 1) 전처리된 CSV 불러오기
df = pd.read_csv("daicwoz_text_phq.csv")

texts = df["text"].astype(str).tolist()
labels = df["phq"].tolist()

print(f"총 데이터 개수: {len(texts)}")

# 2) 한국어 SBERT 사용 (추천 모델)
model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
model = SentenceTransformer(model_name)

print("한국어 SBERT 모델 로드 완료!")

# 3) 임베딩 생성
embeddings = model.encode(
    texts,
    batch_size=8,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("임베딩 생성 완료!")
print("임베딩 shape:", embeddings.shape)

# 4) 저장
np.save("emb_ko_sbert.npy", embeddings)
np.savetxt("phq_labels.txt", labels, fmt="%d")

print("파일 저장 완료!")
print("- emb_ko_sbert.npy")
print("- phq_labels.txt")
