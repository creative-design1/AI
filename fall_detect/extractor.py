import mediapipe as mp
import cv2
import numpy as np


class PoseExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, 
                                           model_complexity=1, 
                                           enable_segmentation=False, 
                                           min_detection_confidence=0.5, 
                                           min_tracking_confidence=0.5)
        # draw = mp.solutions.drawing_utils

    def extract_keypoints(self, frame):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            
            kp = np.zeros((33, 4), dtype=np.float32)
            
            if results.pose_landmarks:
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    kp[i] = [lm.x, lm.y, lm.z, lm.visibility]
                    # visibility가 임계점보다 낮으면 0으로 채우는 방안 (또는 이전 프레임으로 보간하는 방법도 가능)
                    # if  v < visibility_threshold:
                    #     kp[i] = np.array([0, 0, 0, 0], dtype=np.float32)

            flat = kp.reshape(-1)
            
            return flat