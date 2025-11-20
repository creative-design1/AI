import queue, tempfile, os, time, soundfile as sf

USE_FASTER = True
BACKEND = "whisper"  # or "vosk" if you prefer offline vosk

if USE_FASTER:
    try:
        from faster_whisper import WhisperModel
        _FW_AVAILABLE = True
    except Exception:
        _FW_AVAILABLE = False
else:
    _FW_AVAILABLE = False

if _FW_AVAILABLE:
    fw_model = WhisperModel("tiny", device="cpu", compute_type="int8")
else:
    import whisper
    whisper_model = whisper.load_model("tiny")
    
class STT:
    def __init__(self, audio_queue: queue.Queue, text_queue: queue.Queue, sample_rate = 16000):
        self.audio_queue = audio_queue
        self.text_queue = text_queue
        self.sample_rate = sample_rate
        
    def run(self):
        print("STT started")
        while True:
            arr = self.audio_queue.get()
            if arr is None:
                time.sleep(0.1)
                continue
            # write tmp wav
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmpname = tmp.name
            tmp.close()
            sf.write(tmpname, arr, self.sr, subtype='PCM_16')

            text = ""
            try:
                if _FW_AVAILABLE:
                    segments, info = fw_model.transcribe(tmpname, beam_size=5)
                    text = " ".join([seg.text for seg in segments]).strip()
                else:
                    res = whisper_model.transcribe(tmpname)
                    text = res.get("text", "").strip()
            except Exception as e:
                print("STT error:", e)
            finally:
                try: os.remove(tmpname)
                except: pass

            if text:
                print("STT ->", text)
                try:
                    self.text_queue.put_nowait(text)
                except queue.Full:
                    pass