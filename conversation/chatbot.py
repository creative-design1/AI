from .audioRecorder import AudioRecorder
from .tts import TTS
from .stt import STT
from .llm import LLM
import queue
import threading

class ChatBot:
    
    def __init__(self, sender: object = None):
        self.stop_event = threading.Event()
        self.audio = AudioRecorder(queue.Queue())
        self.stt = STT(audio_queue=self.audio.audio_queue, text_queue=queue.Queue(), recorder=self.audio)
        self.llm = LLM(text_queue=self.stt.text_queue, reply_queue=queue.Queue(), recorder=self.audio, sender=sender)
        self.tts = TTS(reply_queue=self.llm.reply_queue, recorder=self.audio)
        self.threads = []
    
    def start(self):
        threads = [
            threading.Thread(target=self.audio.run),
            threading.Thread(target=self.stt.run),
            threading.Thread(target=self.llm.run),
            threading.Thread(target=self.tts.run),
        ]
        
        for t in threads:
            t.start()
        
        print("ChatBot: 모든 내부 컴포넌트 스레드 시작 완료.")
        # 이 메서드는 이제 즉시 메인 스레드로 제어를 반환합니다.
        
        
    def is_alive(self):
        """내부 스레드 중 하나라도 살아있는지 확인합니다."""
        return any(t.is_alive() for t in self.threads)

    def stop(self):
        """내부 스레드를 안전하게 종료합니다."""
        # 각 컴포넌트의 stop() 메서드를 호출하여 내부 루프를 종료합니다.
        self.audio.stop()
        self.stt.stop()
        self.llm.stop() # LLM, STT, TTS에 stop() 메서드가 정의되어 있다고 가정
        self.tts.stop() 
        
        # 스레드 종료 대기
        for t in self.threads:
            t.join(timeout=5)
        print("ChatBot: 모든 내부 스레드 종료 완료.")