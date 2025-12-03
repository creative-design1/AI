import queue
import threading
import time
import tempfile
import os
from gtts import gTTS
from playsound import playsound

class TTS:
    def __init__(self, reply_queue: queue.Queue, recorder: object = None):
        self.queue = reply_queue
        #self.engine = pyttsx3.init()
        self.recorder = recorder  # AudioRecorder 인스턴스 (mute 제어 위해)

        # pyttsx3 설정(옵션)
        #self.engine.setProperty('rate', 150)  # 속도
        #self.engine.setProperty('volume', 1.0)
    """
    def _speak_blocking(self, text):
        # pyttsx3는 같은 프로세스에서 음성 재생 중 블로킹될 수 있으므로
        # 재생 전 recorder를 mute 하고, 재생 후 복원
        
        self.engine.say(text)
        char_count = len(text)
        estimated_time = (char_count / 150) * 40 # 대략적인 안전계수 포함
        min_time = 1
        estimated_time = max(min_time, estimated_time)

        print(f"TTS: Estimated wait time: {estimated_time:.2f}s")
        #time.sleep(estimated_time)

        self.engine.stop()
        
        if self.recorder:
            time.sleep(0.05)
            self.recorder.mute = False

    def run(self):
        print("TTSWorker: started")
        while True:
            text = self.queue.get()
            if not text:
                continue
            print("TTS ->", text)
            try:
                self._speak_blocking(text)
            except Exception as e:
                print("TTS error:", e)
                
            print("speak end")
    """
    
    def _speak(self, text):
        # 임시 mp3 파일 생성
        tmp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        try:
            tts = gTTS(text=text, lang="ko")
            tts.save(tmp_path)
            playsound(tmp_path)
        except Exception as e:
            print("TTS error:", e)
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass
        self.recorder.mute = False
        print("audio run")

    def run(self):
        print("TTSWorker: started")
        while True:
            text = self.queue.get()
            if not text:
                continue
            print("TTS ->", text)
            self._speak(text)