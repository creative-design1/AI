import threading
import queue
import time
from fall_detect.sender import Sender
from depression.depression_model import load_model_assets, process_daily_text


SENDER_URL = None

class depressionProcessor(threading.Thread):
    def __init__(self, depression_queue, url):
        super().__init__()
        self.depression_queue = depression_queue
        self.sender = Sender(url=url)
        self._stop = threading.Event()
        self.daemon = True
        
        self.model, self.tokenizer, self.device = load_model_assets()
        if not self.model:
            print("모델 로드 실패")
            self._stop.set()
        else:
            print("모델 로드 성공")
            
    def run(self):
        if self._stop.is_set():
            return
        while not self._stop.is_set():
            try:
                daily_text_list = self.depression_queue.get(timeout=1)

                print(f"하루 대화 문장 수: {len(daily_text_list)} 문장")
                
                depression_avg_score, _ = process_daily_text(
                    self.model,
                    self.tokenizer,
                    self.device,
                    daily_text_list
                )
                
                data = {
                    "elderyUserId": 1,
                    "depression_score": depression_avg_score
                }
                self.sender.send(data)
                print(f"우울증 전송 완료 우울증 점수: {depression_avg_score:.4f}")
            
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"error: {e}")
                
    def stop(self):
        self._stop.set()