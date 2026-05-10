import numpy as np
from src.models.model_lstm import GAIT_LANDMARK_INDICES

def hip_center_keypoints(kp: np.ndarray) -> np.ndarray:
    mid_hip_x = (kp[:, 23*2] + kp[:, 24*2]) / 2
    mid_hip_y = (kp[:, 23*2+1] + kp[:, 24*2+1]) / 2

    out = np.zeros((kp.shape[0], len(GAIT_LANDMARK_INDICES)*2), dtype=np.float32)

    for i, lm in enumerate(GAIT_LANDMARK_INDICES):
        out[:, i*2]   = kp[:, lm*2]   - mid_hip_x
        out[:, i*2+1] = kp[:, lm*2+1] - mid_hip_y

    max_abs = np.max(np.abs(out))
    if max_abs > 1e-6:
        out /= max_abs

    return out