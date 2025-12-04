import queue, tempfile, os, time, soundfile as sf
import numpy as np
import webrtcvad

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
    fw_model = WhisperModel("small", device="cpu", compute_type="int8")
else:
    import whisper
    whisper_model = whisper.load_model("small")
    
class STT:
    def __init__(self, audio_queue: queue.Queue, text_queue: queue.Queue, recorder: object = None, sample_rate = 16000):
        self.audio_queue = audio_queue
        self.text_queue = text_queue
        self.sample_rate = sample_rate
        self.recorder = recorder
        
        self.vad = webrtcvad.Vad(3)  # 민감도 높임
        self.frame_ms = 20
        self.frame_size = int(self.sample_rate * self.frame_ms / 1000)

        self.buffer = []
        self.silence_count = 0
        self.speech_count = 0
        
        self.silence_limit = int(0.7 / (self.frame_ms / 1000))  # 1초 침묵이면 문장 끝
        
        self.min_length_samples = int(sample_rate * 0.5)
        
        self.amplitude_threshold = 0.01
        
    def is_speech(self, pcm16):
        if len(pcm16) < self.frame_size:
            return False
        frame = pcm16[:self.frame_size]

        return self.vad.is_speech(frame.tobytes(), self.sample_rate)
        
    def run(self):
        print("STT started")

        while True:
            arr = self.audio_queue.get()
            if arr is None:
                time.sleep(0.1)
                continue
            
            pcm16 = (arr * 32768).astype(np.int16)
            self.buffer.append(pcm16)
            
            if self.is_speech(self.buffer):
                self.speech_count += 1
                self.silence_count = 0
            else:
                self.silence_count += 1
            
            total_samples = sum(len(chunk) for chunk in self.buffer)
            
            if self.silence_count >= self.silence_limit and total_samples >= self.min_length_samples:
            
                merged = np.concatenate(self.buffer)
                self.buffer = []
                self.silence_count = 0
                self.speech_count = 0
                
                if np.max(np.abs(merged)) < self.amplitude_threshold:
                    continue
                
                # write tmp wav
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)  
                tmpname = tmp.name
                tmp.close()
                sf.write(tmpname, merged, self.sample_rate, subtype='PCM_16')

                text = ""
                try:
                    if _FW_AVAILABLE:
                        segments, info = fw_model.transcribe(
                            tmpname,
                            beam_size=5,
                            temperature=0.0,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500),
                            language = "ko",
                            condition_on_previous_text = False
                        )
                        text = " ".join([seg.text for seg in segments]).strip()
                    else:
                        res = whisper_model.transcribe(
                            tmpname,
                            temperature=0.0,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500),
                            language = "ko",
                            condition_on_previous_text = False
                        )
                        text = res.get("text", "").strip()
                except Exception as e:
                    print("STT error:", e)
                finally:
                    try: os.remove(tmpname)
                    except: pass

                if text and len(text) > 1:
                    self.recorder.set_mute(True)
                    print("STT ->", text)
                    try:
                        self.text_queue.put_nowait(text)
                    except queue.Full:
                        pass