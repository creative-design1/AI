import queue, time
USE_OPENAI_API = True  # True: OpenAI API 사용, False: HuggingFace local
if USE_OPENAI_API:
    import openai
    OPENAI_API_KEY = None
    openai.api_key = OPENAI_API_KEY
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    model_name = None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to("cpu")

class LLM:
    def __init__(self, text_queue: queue.Queue, reply_queue: queue.Queue):
        self.text_queue = text_queue
        self.reply_queue = reply_queue

    def run(self):
        print("LLMWorker: started")
        while True:
            text = self.text_queue.get()
            if not text:
                continue
            try:
                if USE_OPENAI_API:
                    # 간단한 completion 예시 (gpt-4o-mini 등으로 교체 가능)
                    resp = openai.ChatCompletion.create(
                        model="gpt-4o-mini", # 또는 gpt-4o, gpt-4o-mini 등
                        messages=[{"role":"user","content": text}],
                        max_tokens=150,
                        temperature=0.6
                    )
                    reply = resp["choices"][0]["message"]["content"].strip()
                else:
                    prompt = text + "\n\n간단히 대답해줘:"
                    inputs = tokenizer(prompt, return_tensors="pt")
                    outputs = model.generate(**inputs, max_new_tokens=150)
                    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                print("LLM error:", e)
                reply = "죄송합니다. 지금 답변을 생성할 수 없습니다."

            print("LLM ->", reply)
            try:
                self.reply_queue.put_nowait(reply)
            except queue.Full:
                pass