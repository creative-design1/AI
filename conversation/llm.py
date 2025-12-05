import ollama
import queue
import time

class LLM:
    def __init__(self, text_queue: queue.Queue, reply_queue: queue.Queue, recorder: object = None, sender: object = None):
        self.text_queue = text_queue
        self.reply_queue = reply_queue
        self.recorder = recorder
        self.sender = sender
        
    def run(self):
        print("LLMWorker: started")
        
        while True:
            text = self.text_queue.get()
            if not text:
                continue
            
            try:
                data = {
                    "elderyUserId": 1,
                    "text": text,
                    "isUser": True,
                    "timestamp": "2025-12-05"
                }
                self.sender.send(data)
                text = text + "두 문장 이내로 말해주고 이모티콘 등 특수문자 넣지마"
                time.sleep(0.05)
                result = ollama.generate(model="gemma3:4b", prompt=text)
                reply = result.response
                self.reply_queue.put_nowait(reply)
                print("LLM ->", reply)
            except Exception as e:
                self.recorder.set_mute(False)
                print("LLM error: ", e)