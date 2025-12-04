import threading
import queue
import time
import numpy as np
import soundfile as sf
from conversation.audioRecorder import AudioRecorder
from conversation.stt import STT
from conversation.llm import LLM
from conversation.tts import TTS
import subprocess

#url = "http://10.93.152.178:8554/audio_stream"

def play_m4a_to_queue(audio_queue, m4a_path, sample_rate=16000):
    """
    m4a 파일을 ffmpeg로 디코딩하여 AudioRecorder와 동일한 방식으로
    raw PCM 데이터(float32, -1~1 범위)를 queue에 넣어주는 함수
    """
    cmd = [
        "ffmpeg",
        "-i", m4a_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "pipe:1"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=4096
    )

    print("[M4A-Player] Start feeding audio…")

    chunk_size = 640   # AudioRecorder와 동일 (0.2초 분량)

    while True:
        data = process.stdout.read(chunk_size)
        if not data:
            break

        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            audio_queue.put_nowait(audio)
        except queue.Full:
            pass

        time.sleep(0.2)  # 실제 스트리밍처럼 처리

    print("[M4A-Player] Finished feeding audio.")


def main():
    print("Conversation System Starting...")
    
    audio_queue = queue.Queue()
    text_queue = queue.Queue()
    reply_queue = queue.Queue()
    
    recorder = AudioRecorder(audio_queue)
    stt_worker = STT(audio_queue, text_queue, recorder)
    llm_worker = LLM(text_queue, reply_queue, recorder)
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
    #feeder = threading.Thread(target=play_m4a_to_queue, args=(audio_queue, "stt_test.m4a"), daemon=True)
    #feeder.start()
    
    time.sleep(3)
    reply_queue.put("안녕하세요. 테스트 음성 출력입니다.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        recorder.stop()
        time.sleep(1)
    
if __name__ == "__main__":
    main()