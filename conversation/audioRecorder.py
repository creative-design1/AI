import subprocess
import numpy as np
import threading
import time

class AudioRecorder:
    def __init__(self, audio_queue, rtsp_url, sample_rate=16000):
        self.audio_queue = audio_queue
        self.rtsp_url = rtsp_url
        self.sample_rate = sample_rate
        self._stop = threading.Event()

    def run(self):
        cmd = [
            "ffmpeg",
            "-i", self.rtsp_url,
            "-vn",
            "-ac", "1",
            "-ar", str(self.sample_rate),
            "-f", "s16le",
            "pipe:1"
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=4096)

        print("[RTSPAudioFeeder] Started. Listening...")

        while not self._stop.is_set():
            data = process.stdout.read(3200)  # 0.2초 분량
            if not data:
                continue

            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            try:
                self.audio_queue.put_nowait(audio)
            except:
                pass

        process.kill()

    def stop(self):
        self._stop.set()