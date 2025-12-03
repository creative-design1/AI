import time
import cv2
from pathlib import Path
from extractor import PoseExtractor
from sender import Sender
from fall import Fall_Detector
from features import compute_features
from walking import WalkingDetector
#from chatbot import ChatBot

BASE_DIR = Path.cwd().parent

model_path = BASE_DIR / "models" / "best_fall_model.pth"
scaler_path = BASE_DIR / "src" / "scaler.save"
video_source = 0
url = "http://192.168.1.50:8080"
fall_api_path = url + "/api/events/fall-detection"
stride_api_path = url + "/api/events/features"
audio_source = None #url + "/audio.opus"

detector = Fall_Detector(model_path=model_path, scaler_path=scaler_path, device='cpu')
extractor = PoseExtractor()
#fall_sender = Sender(url=fall_api_path)
#stride_sender = Sender(url=stride_api_path)
walk = WalkingDetector()
#chatbot = ChatBot(rstp_url=audio_source)

cap = cv2.VideoCapture("http://10.93.152.178:8080/?action=stream")

if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit(1)
    
print("starting fall detection!")

#chatbot.start()
#print("ChatBot started.")

sent_fall = False
missing_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from video source.")
        break
    
    features = None
    keypoints = extractor.extract_keypoints(frame)
    if not walk.detect_person(keypoints) or keypoints is None:
        #print("No person detected.")
        missing_count += 1
        if missing_count >= 10:
            detector.buffer.clear()
            walk.reset()
            sent_fall = False
            
        continue
    else:
        missing_count = 0
        #print("Person detected.")
        
    walking = walk.update(keypoints)
    detector.update_sequence(keypoints)
    
    fall_detected, fall_prob = detector.predict()
    
    #if len(detector.buffer) == detector.buffer.maxlen:
    #    features = compute_features(list(detector.buffer), fps=30)
    #else:
    #    features = {}
    """
    data = {
            "elderlyUserId": 1,
            "fall_prob": fall_prob,
            "fall_detected": fall_detected,
            #"detectedAt": "2024-06-01 12:00:00"
            "detectedAt": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }
    """
    #print(data)
    if fall_detected:
        if not sent_fall:
            data = {
                "elderlyUserId": 1,
                "fall_prob": fall_prob,
                "fall_detected": fall_detected,
                "detectedAt": "2024-06-01 12:00:00"
                #"detectedAt": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            }
                #"stride_mean": features.get("stride_mean", 0.0),
                #"stride_std": features.get("stride_std", 0.0),
                #"velocity": features.get("velocity", 0.0),
            #fall_sender.send(data)
            print(f"Fall detected! Probability: {fall_prob:.2f}, Data sent.")
            sent_fall = True
        
        walk.reset()
        
    else:
        sent_fall = False

    if not fall_detected and walking is not None:
        features = compute_features(list(walking), fps=30)
        print(features)
        features = {
            "elderyUserId": 1,
            "stride_mean": features["stride_mean"],
            "stride_std": features["stride_std"],
            "velocity": features["velocity"]
        }
        #stride_sender.send(features)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
print("Fall detection stopped.")