import subprocess
import numpy as np
import threading
import queue
import os
import time

"""
class AudioRecorder:
    def __init__(self, audio_queue, rtsp_url, sample_rate=16000):
        self.audio_queue = audio_queue
        self.rtsp_url = rtsp_url
        self.sample_rate = sample_rate
        self._stop = threading.Event()
        self.mute = False

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

        # stderr을 PIPE로 해야해서 ffmpeg 출력 끊기지 않음
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=4096
        )

        print("[AudioRecorder] Started streaming…")

        while not self._stop.is_set():
            data = process.stdout.read(3200)  # 0.2초 분량
            if not data:
                continue

            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            if self.mute:
                audio = None
                continue
                
            try:
                self.audio_queue.put_nowait(audio)
            except queue.Full:
                pass

        process.kill()
        print("[AudioRecorder] Stopped.")

    def stop(self):
        self._stop.set()
"""
import socket
import threading
import queue
import numpy as np

# AudioRecorder 클래스를 UDP 수신 리스너로 재정의합니다.
class AudioRecorder(threading.Thread):
    def __init__(self, audio_queue, bind_port=8554, sample_rate=16000):
        super().__init__()
        self.audio_queue = audio_queue
        # 노트북의 UDP 포트
        self.bind_port = bind_port
        self.sample_rate = sample_rate
        self.frame_size = 320
        self._stop = threading.Event()
        self.daemon = True
        self.mute = False
        
        self._temp_buffer = b''
    def clear_queue(self):
        """큐를 비우는 헬퍼 메서드"""
        while True:
            try:
                # 항목을 가져옵니다. 큐가 비어있으면 예외가 발생합니다.
                self.audio_queue.get_nowait()
            except queue.Empty:
                # 큐가 비었으므로 루프를 종료합니다.
                break

    def run(self):
        min_bytes = self.frame_size * 2
        # UDP 소켓 생성 및 포트 바인딩
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # 0.0.0.0 주소의 8554 포트를 열고 수신 대기
            s.bind(('0.0.0.0', self.bind_port))
            print(f"[UDP-RX] UDP Listener started on port {self.bind_port}...")
            
            while not self._stop.is_set():
                long = False
                try:
                    # UDP 데이터 수신 (4096 bytes 청크)
                    data, addr = s.recvfrom(4096)
                    
                    if data:
                        if self.mute:
                            print("mute = true")
                            data = None
                            continue
                        # 데이터 수신 성공! NumPy 변환 및 큐에 삽입 (기존 로직 유지)
                        
                        self._temp_buffer += data
                        
                        #print("오디오레코더" + len(self._temp_buffer))
                        while len(self._temp_buffer) >= min_bytes:
                            long = False
                            chunk_bytes = len(self._temp_buffer)
                            if len(self._temp_buffer) >= 4096:
                                chunk_data = self._temp_buffer[:4096]
                                self._temp_buffer = self._temp_buffer[4096:]
                                long = True
                            else:
                                # 4096 bytes를 확보하지 못하면 다음 수신을 기다림
                                long = False
                                break
                            
                            audio = np.frombuffer(chunk_data, dtype=np.int16).astype(np.float32) / 32768.0
                        if not long:
                            continue
                        try:
                            self.audio_queue.put_nowait(audio)
                        except queue.Full:
                            pass
                except Exception as e:
                    print(f"[UDP-RX] Error receiving data: {e}")
                    break
        print("[UDP-RX] UDP Listener stopped.")
        
    def set_mute(self, mute):
        self.mute = mute
        print("뮤트 설정: ", mute)
        if mute:
            self.clear_queue()
        
    def stop(self):
        self._stop.set()
# 메인 코드에서 실행:
# recorder = AudioRecorder(audio_queue=audio_queue, bind_port=8554)
# recorder.start()