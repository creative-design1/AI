import threading
import queue
import time

from conversation.audioRecorder import AudioRecorder
from conversation.stt import STT
from conversation.llm import LLM
from conversation.tts import TTS


url = "http://192.168.0.9:8080/audio.opus"
def main():
    print("Conversation System Starting...")
    
    audio_queue = queue.Queue()
    text_queue = queue.Queue()
    reply_queue = queue.Queue()
    
    recorder = AudioRecorder(audio_queue, url)
    stt_worker = STT(audio_queue, text_queue)
    llm_worker = LLM(text_queue, reply_queue)
    tts_worker = TTS(reply_queue, recorder=recorder)
    
    threads = [
    threading.Thread(target=recorder.run),
    threading.Thread(target=stt_worker.run),
    threading.Thread(target=llm_worker.run),
    threading.Thread(target=tts_worker.run),
    ]
    
    for t in threads:
        t.daemon = True
        t.start()
    
    time.sleep(3)
    reply_queue.put("안녕하세요. 테스트 음성 출력입니다.")
    
    
if __name__ == "__main__":
    main()