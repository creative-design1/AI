from audioRecorder import AudioRecorder
from tts import TTS
from stt import STT
from llm import LLM
import queue
import threading

class ChatBot:
    
    def __init__(self):
        self.stop_event = threading.Event()
        self.audio = AudioRecorder(queue.Queue(), stop_event=self.stop_event)
        self.stt = STT(audio_queue=self.audio.audio_queue, text_queue=queue.Queue(), recorder=self.audio, stop_event=self.stop_event)
        self.llm = LLM(text_queue=self.stt.text_queue, reply_queue=queue.Queue(), recorder=self.audio, stop_event=self.stop_event)
        self.tts = TTS(reply_queue=self.llm.reply_queue, recorder=self.audio, stop_event=self.stop_event)
    
    def start(self):
        threads = [
            threading.Thread(target=self.audio.run),
            threading.Thread(target=self.stt.run),
            threading.Thread(target=self.llm.run),
            threading.Thread(target=self.tts.run),
        ]
        
        for t in threads:
            t.start()
        
        try:
            for t in threads:
                t.join()
                
        except KeyboardInterrupt:
            print("ChatBot stopped.")
            self.stop_event.set()
            for t in threads:
                t.join()