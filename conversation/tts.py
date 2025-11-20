import pyttsx3, queue, time

class TTS:
    def __init__(self, reply_queue: queue.Queue, recorder: object = None):
        self.queue = reply_queue
        self.engine = pyttsx3.init()
        self.recorder = recorder  # AudioRecorder 인스턴스 (mute 제어 위해)

        # pyttsx3 설정(옵션)
        self.engine.setProperty('rate', 150)  # 속도
        self.engine.setProperty('volume', 1.0)

    def _speak_blocking(self, text):
        # pyttsx3는 같은 프로세스에서 음성 재생 중 블로킹될 수 있으므로
        # 재생 전 recorder를 mute 하고, 재생 후 복원
        if self.recorder:
            self.recorder.muted = True
            time.sleep(0.05)  # ensure mic is muted

        self.engine.say(text)
        self.engine.runAndWait()

        if self.recorder:
            time.sleep(0.05)
            self.recorder.muted = False

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