import threading, queue, time, requests, json
from datetime import datetime

"""
data format:
{
    "device_id": string,
    "fall_prob": float,
    "fall_detected": bool,
    "timestamp": "string",
}
"""
"""
{
    "device_id": string,
    "stride_mean": float,
    "stride_std": float,
    "velocity": float,
    "timestamp": "string",
}
"""


class Sender:
    
    def __init__(self, url, api_key=None, max_queue=1000):
        self.url = url
        self.api_key = api_key
        self.queue = queue.Queue(max_queue)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._send_loop, daemon=True)
        self.thread.start()
        
        
    def send(self, data):
        try:
            self.queue.put_nowait({"data": data, "attempts": 0 })
            
        except queue.Full:
            
            return False
        
        
    def _send_loop(self):
        print("Sender thread started")
        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            attempts = item["attempts"]
            success = False
            
            while attempts < 5 and not success:
                attempts += 1
                success = self._post(item)
                if not success:
                    sleep = min(2 ** (attempts - 1), 30)
                    time.sleep(sleep)
                    
            if not success:
                self._persist_failure(item)
            self.queue.task_done()
            
            
    def _post(self, item):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        try:
            r = requests.post(self.url, json = item["data"], headers=headers, timeout=5)
            print("data sent to server")
            r.raise_for_status()
            return True
        except Exception as e:
            print(f"Error sending data: {e}")
            return False
    
    def _persist_failure(self, item):
        ts = datetime.utcnow().isoformat().replace(":", "-")
        fname = f"failed_send_{ts}.json"
        with open(fname, "w") as f:
            json.dump(item, f)
            
    def stop(self):
        self._stop.set()
        self.thread.join(timeout=2)