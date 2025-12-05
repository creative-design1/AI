import numpy as np
from collections import deque
from .extractor import PoseExtractor

class WalkingDetector:
    def __init__(self):
        self.prev_keypoints = None
        self.buffer = []
        self.walking = False
        self.walk_start_counter = 0
        self.walk_end_counter = 0
        self.missing_counter = 0
        self.smooth_factor = 0.0
        
    def detect_person(self, kp):
        vis = kp.reshape(33, 4)[:, 3]
        return np.sum(vis > 0.3) >= 12
    
    def diff(self, kp):
        
        prev = self.prev_keypoints.reshape(33, 4)
        curr = kp.reshape(33, 4)
        
        ankle_prev = (prev[27][:3] + prev[28][:3]) / 2
        ankle_curr = (curr[27][:3] + curr[28][:3]) / 2
        
        return np.linalg.norm(ankle_curr - ankle_prev)
    
    def update(self, kp):
        if not self.detect_person(kp):
            
            self.missing_counter += 1
            
            if self.walking:
                if self.missing_counter <= 10:
                    return None
                else:
                    return self._end_walking()

            self.prev_keypoints = None
            return None
        
        self.missing_counter = 0
        
        if self.prev_keypoints is None:
            self.prev_keypoints = kp
            return None
        
        abs_diff = self.diff(kp)
        self.smooth_factor = 0.9 * self.smooth_factor + 0.1 * abs_diff
        diff = self.smooth_factor
        
        self.prev_keypoints = kp
        
        is_step = diff > 0.03
        
        if is_step:
            self.walk_start_counter += 1
            if not self.walking:
                if self.walk_start_counter >= 4:
                    self.walking = True
                    self.buffer.clear()
                    self.walk_end_counter = 0
        else:
            self.walk_start_counter = 0
            
        
        
        if self.walking:
            self.buffer.append(kp)
            if not is_step:
                self.walk_end_counter += 1
            else:
                self.walk_end_counter = 0
                
            if self.walk_end_counter >= 10:
                return self._end_walking()
            
        return None
    
    def _end_walking(self):
        self.walking = False
        self.walk_start_counter = 0
        self.walk_end_counter = 0
        self.missing_counter = 0
        self.smooth_factor = 0.0
        self.prev_keypoints = None
        
        if len(self.buffer) < 10:
            self.buffer.clear()
            return None
        
        saved = self.buffer.copy()
        self.buffer.clear()
        return saved
        
    def reset(self):
        self.prev_keypoints = None
        self.buffer.clear()
        self.walking = False
        self.walk_start_counter = 0
        self.walk_end_counter = 0
        self.missing_counter = 0
        self.smooth_factor = 0.0