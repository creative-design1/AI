import numpy as np


def stride_length(seq):
    l_ankle = seq[:, 27, :]
    r_ankle = seq[:, 28, :]
    foot_dist = np.linalg.norm(l_ankle - r_ankle, axis=1)
    stride_mean = np.mean(foot_dist)
    stride_std = np.std(foot_dist)
    
    return stride_mean, stride_std
    
    pass


def stride_velocity(seq, fps):
    l_hip = seq[:, 23, :]
    r_hip = seq[:, 24, :]
    center = (l_hip + r_hip) / 2.0
    
    dist = np.linalg.norm(np.diff(center, axis=0), axis=1)
    avg_vel = np.mean(dist) * fps
    return avg_vel


def compute_features(seq, fps):
    
    seq = np.array(seq, dtype=np.float32)
    seq = seq.reshape(len(seq), 33, 4)
    seq = seq[:, : , :3]
    
    mean_length, std_length = stride_length(seq)
    velocity = stride_velocity(seq, fps)
    
    return {
        "stride_mean": float(mean_length),
        "stride_std": float(std_length),
        "velocity": float(velocity)
    }