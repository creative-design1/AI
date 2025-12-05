
import websocket
import json
import threading
import queue


class SpringData:
    
    def __init__(self, reply_queue, depression_queue, url):
        self.reply_queue = reply_queue
        self.depression_queue = depression_queue
        self.ws = None
        self.thread = None
        self.url = url
        
    def on_message(self, ws, message):
        received_data = json.loads(message)
        if "reminders" in received_data:
            print(received_data)
            self.reply_queue.put_nowait(received_data["reminders"][1])
        else:
            daily_text = received_data["text"]
            self.depression_queue.put_nowait(daily_text)

    def on_error(self, ws, error):
        print("웹소켓 에러: ", error)
        
    def on_close(self, ws, close_status_code, close_msg):
        print(f"웹소켓 종료: [{close_status_code}] {close_msg}")
        
    def on_open(self, ws):
        print("웹소켓 연결")
        
    def run_websocket_client(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open = self.on_open,
            on_message = self.on_message,
            on_error = self.on_error,
            on_close = self.on_close
        )
        self.ws.run_forever(ping_interval=30, ping_timeout=10)
        
    def start(self):
        self.thread = threading.Thread(target=self.run_websocket_client)
        self.thread.daemon = True
        self.thread.start()
        print("웹소켓 쓰레드 시작")