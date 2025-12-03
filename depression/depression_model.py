import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import os

# --- 설정값은 학습 시와 동일해야 합니다 ---
MODEL_NAME = "xlm-roberta-base"
NUM_LABELS = 2
MODEL_PATH = "xlm_roberta_depression_model.bin"
MAX_LEN = 128
# ----------------------------------------

def predict_depression_score(text, model, tokenizer, device, max_len=MAX_LEN):
    """
    주어진 텍스트에 대한 우울증 확률(레이블 1)을 예측합니다.
    """
    model.eval()
    
    # 1. 텍스트 인코딩 (학습 시와 동일한 방식으로)
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens = True,
        max_length = max_len,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        return_tensors = 'pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 2. 예측 수행
    with torch.no_grad():
        outputs = model(input_ids, attention_mask = attention_mask)
        
        # 3. 확률 변환 및 점수 추출
        # F.softmax를 사용하여 로짓(logits)을 확률로 변환합니다.
        probabilities = F.softmax(outputs.logits, dim=1)
        
        # 레이블 1 (우울증)의 확률을 가져옵니다.
        depression_prob = probabilities[0][1].item()
        
    return depression_prob

def load_model_assets():
    """
    모델, 토크나이저 및 장치 설정을 한 번만 수행하고 반환합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # 1. 토크나이저 로드
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    print(f"토크나이저 로드 완료: {MODEL_NAME}")
    # 1. 현재 파일(depression_model.py)의 디렉토리 경로를 가져옵니다.
    # __file__ 변수는 현재 실행되는 모듈의 경로를 담고 있습니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 모델 파일의 전체 경로(절대 경로)를 생성합니다.
    model_full_path = os.path.join(current_dir, MODEL_PATH)
    # 2. 모델 구조 로드
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = NUM_LABELS
    )

    # 3. 가중치 로드
    try:
        model.load_state_dict(torch.load(model_full_path, map_location=device))
        print(f"모델 가중치 로드 완료: {model_full_path}")
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{model_full_path}'을(를) 찾을 수 없습니다.")
        return None, None, None # 실패 시 None 반환

    # 4. 모델 설정
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def process_daily_text(model, tokenizer, device, daily_texts):
    """
    하루치 문장 목록을 받아 각 문장의 우울증 확률을 계산하고 평균을 냅니다.
    """
    if model is None or tokenizer is None or device is None:
        print("모델 로드에 실패하여 예측을 시작할 수 없습니다.")
        return None, []

    all_scores = []
    
    print("\n--- 하루치 문장 예측 시작 ---")
    for i, text in enumerate(daily_texts):
        # 1. 로드된 모델 객체를 사용하여 예측 함수 호출
        score = predict_depression_score(text, model, tokenizer, device)
        all_scores.append(score)
        print(f"[{i+1}/{len(daily_texts)}] '{text[:30]}...' -> 점수: {score:.4f}")
    
    # 2. 결과 평균 계산
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        print(f"\n총 {len(all_scores)}개 문장의 평균 우울증 점수: {avg_score:.4f}")
        return avg_score, all_scores
    else:
        return 0, []

"""
# A. 프로그램 시작 시 모델과 토크나이저를 한 번만 로드
    model, tokenizer, device = load_model_assets()

    # B. 하루치 텍스트 데이터 (실제 데이터로 대체)
    daily_texts = [
        "나는 매일 밤 잠을 설친다. 모든 게 의미 없고, 무기력한 기분이다.",
        "오늘은 오랜만에 친구들과 즐거운 시간을 보냈다. 맛있는 음식도 먹고 행복했어.",
        "삶이 너무 힘들다. 아무도 나를 이해하지 못하는 것 같다.",
        "점심으로 김치찌개를 먹었는데 정말 맛있었다.",
        "다시 시작할 힘이 없다. 그냥 모든 걸 포기하고 싶다."
    ]

    # C. 로드된 객체를 사용하여 예측 로직 실행
    average_depression_score, all_scores = process_daily_text(model, tokenizer, device, daily_texts)

    # D. 최종 결과 출력
    if average_depression_score is not None:
        print("\n--- 최종 결과 ---")
        print(f"하루 우울증 지수: {average_depression_score:.4f}")
        """