import pandas as pd
from sklearn.model_selection import train_test_split
print("import torch")
import torch
print("import torch, dataloader")
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import get_linear_schedule_with_warmup
print("transformer")
from torch.optim import AdamW
print("adam")
import torch.nn.functional as F
print("F")
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

def train():
    df = pd.read_csv("reddit_depression_dataset.csv")

    df['title'] = df['title'].fillna('')
    df['body'] = df['body'].fillna('')

    df['text'] = df['title'] + ' [SEP] ' + df['body']

    df = df.dropna(subset=['label'])
    df = df[['text', 'label']]

    df['label'] = df['label'].astype(int)

    train_df, test_df = train_test_split(
        df, test_size = 0.1, random_state = 42, stratify = df['label'])
        
    print(f"총 데이터 수: {len(df)}")
    print(f"학습 데이터 수: {len(train_df)}")
    print(f"테스트 데이터 수: {len(test_df)}")

    MODEL_NAME = "xlm-roberta-base"
    NUM_LABELS = 2

    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = NUM_LABELS
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"사용 장치: {device}")

    class RedditDepressionDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
            
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts.iloc[idx])
            label = self.labels.iloc[idx]
            
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens = True,
                max_length = self.max_len,
                padding = 'max_length',
                truncation = True,
                return_attention_mask = True,
                return_tensors = 'pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
        
    train_dataset = RedditDepressionDataset(
        train_df['text'], train_df['label'], tokenizer
    )
    test_dataset = RedditDepressionDataset(
        test_df['text'], test_df['label'], tokenizer
    )

    BATCH_SIZE = 16

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False
    )

    EPOCHS = 3
    LEARNING_RATE = 2e-5

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    def evaluate(model, data_loader, device):
        model.eval()
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                )
                _, preds = torch.max(outputs.logits, dim=1)
                
                correct_predictions += torch.sum(preds == labels)
                total_samples += labels.size(0)
                
        accuracy = correct_predictions.double() / total_samples
        return accuracy.item()

    print("train start")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS}, Step {step + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_dataloader)
        test_accuracy = evaluate(model, test_dataloader, device)
        
        print(f"\n[Epoch {epoch + 1} 완료]")
        print(f"평균 훈련 손실: {avg_train_loss:.4f}, 테스트 정확되 {test_accuracy:.4f}\n")
        
    torch.save(model.state_dict(), "xlm_roberta_depression_model.bin")
    print("모델 저장 완료")

    def predict_depression_score(text, model, tokenizer, device, max_len=128):
        model.eval()
        
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
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask = attention_mask)
            probabilities = F.softmax(outputs.logits, dim=1)
            
            depression_prob = probabilities[0][1].item()
            
        return depression_prob

    test_text_ko_depressed = "요즘 아무것도 하고 싶지 않고, 잠만 계속 자고싶다. 뭘 해도 기쁘지 않다"
    test_text_ko_normal = "오늘 날씨가 정말 좋아서 기분이 상쾌하다. 밖에 나가서 친구랑 커피 마셨다"

    score_depressed = predict_depression_score(test_text_ko_depressed, model, tokenizer, device)
    score_normal = predict_depression_score(test_text_ko_normal, model, tokenizer, device)

    print(f"\n--- 한국어 문장 예측 결과 (우울증 확률) ---")
    print(f"'{test_text_ko_depressed}' -> 점수: {score_depressed:.4f}")
    print(f"'{test_text_ko_normal}' -> 점수: {score_normal:.4f}")

if __name__ == "__main__":
    train()