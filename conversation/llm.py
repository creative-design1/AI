import ollama
import queue
import time

class LLM:
    def __init__(self, text_queue: queue.Queue, reply_queue: queue.Queue, recorder: object = None):
        self.text_queue = text_queue
        self.reply_queue = reply_queue
        self.recorder = recorder

    def run(self):
        print("LLMWorker: started")
        
        while True:
            text = self.text_queue.get()
            if not text:
                continue
            
            try:
                text = text + "두 문장 이내로 말해주고 이모티콘 등 특수문자 넣지마"
                time.sleep(0.05)
                result = ollama.generate(model="gemma3:4b", prompt=text)
                reply = result.response
                self.reply_queue.put_nowait(reply)
                print("LLM ->", reply)
            except Exception as e:
                print("LLM error: ", e)