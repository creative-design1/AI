import time
import cv2
from pathlib import Path
from extractor import PoseExtractor
from sender import Sender
from fall import Fall_Detector
from features import compute_features
from walking import WalkingDetector

BASE_DIR = Path.cwd().parent

model_path = BASE_DIR / "models" / "best_fall_model.pth"
scaler_path = BASE_DIR / "src" / "scaler.save"
video_source = 0
url = None #"http://172.30.1.37:8080"
api_path = None #url + "/api/events/fall-detection"

detector = Fall_Detector(model_path=model_path, scaler_path=scaler_path, device='cpu')
extractor = PoseExtractor()
sender = Sender(url=api_path)
walk = WalkingDetector()

cap = cv2.VideoCapture("http://192.168.0.3:8080/video")

if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit(1)
    
print("starting fall detection!")

sent_fall = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from video source.")
        break
    
    features = None
    keypoints = extractor.extract_keypoints(frame)
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
            #sender.send(data)
            print(f"Fall detected! Probability: {fall_prob:.2f}, Data sent.")
            sent_fall = True
        
        walk.reset()
        
    else:
        sent_fall = False

    if not fall_detected and walking is not None:
        features = compute_features(list(walking), fps=30)
        print(features)
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
print("Fall detection stopped.")