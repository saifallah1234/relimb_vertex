import json
import sys
import torch
import numpy as np
import pandas as pd  
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os

CLOUD_DATA_DIR = os.environ.get("VERTEX_DATA_DIR", None)

if CLOUD_DATA_DIR:
    PROJECT_ROOT = Path(CLOUD_DATA_DIR).parent  # /gcs/YOUR_BUCKET/relimb
    DATA_ROOT = Path(CLOUD_DATA_DIR)            # /gcs/YOUR_BUCKET/relimb/data
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_ROOT = PROJECT_ROOT / "data"
    
from src.features.preprocessing import hip_center_keypoints
from src.models.model_lstm import GAIT_LANDMARK_INDICES

def filter_gait_keypoints(kp_array):
    """
    Slices the full 66-feature array down to just the specific features 
    required by the LSTM, ignoring the empty face/hands.
    """
    out = np.zeros((kp_array.shape[0], len(GAIT_LANDMARK_INDICES) * 2), dtype=np.float32)
    for i, lm in enumerate(GAIT_LANDMARK_INDICES):
        out[:, i*2]   = kp_array[:, lm*2]
        out[:, i*2+1] = kp_array[:, lm*2+1]
    return out
# ---> ADDED CLEANING FUNCTION <---
def clean_keypoints(kp_array):
    """
    Fills NaN values in a (Frames, 66) array to prevent LSTM gradient explosion.
    1. Interpolates missing frames in the middle (draws a line between known points).
    2. Backward fills NaNs at the start (simulates person standing still before entering).
    3. Forward fills NaNs at the end.
    4. Fills with 0.0 only as an absolute last resort.
    """
    df = pd.DataFrame(kp_array)
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.bfill()
    df = df.ffill()
    df = df.fillna(0.0)
    return df.to_numpy(dtype=np.float32)


class ProGaitDataset(Dataset):
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.sessions_dir = DATA_ROOT / "sessions"
        if not self.sessions_dir.exists():
            self.sessions_dir = DATA_ROOT / "session"
        self.index_file = DATA_ROOT / "raw_videos" / "hf" / "dataset_index.json"
        self.mapping_file = DATA_ROOT / "class_mapping.json"

        # Load mappings
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            self.class_mapping = json.load(f)

        with open(self.index_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Build lookup
        self.meta_lookup = {}
        for item in metadata:
            raw_id = item["ID"]
            clean_id = raw_id.replace(".mp4", "").replace(".avi", "")
            self.meta_lookup[clean_id] = item

        self.valid_sessions = []

        if not self.sessions_dir.exists():
            raise FileNotFoundError("Could not find data/sessions or data/session")

        print("\nScanning session folders...")

        for folder in self.sessions_dir.iterdir():
            if not folder.is_dir() or "_clip_" not in folder.name:
                continue

            kp_path = folder / "keypoints_normalized.npy"
            if not kp_path.exists():
                continue

            parts = folder.name.split("_clip_")[0]
            clean_id = parts.replace("inside_", "").replace("outside_", "")

            meta = self.meta_lookup.get(clean_id, None)

            if meta is None:
                issue_text = "Unknown / Other"
                ccc_score = 0.0
                metrics = np.zeros(6, dtype=np.float32)
            else:
                issue_text = meta.get("clean_primary_issue") or "Unknown / Other"
                ccc_score = float(meta.get("ccc_score") or 0.0)

                # ✔ REAL METRICS (NO per-sample normalization)
                metrics = np.array([
                    meta.get("cadence_bpm", 0.0),
                    meta.get("step_count", 0.0),
                    meta.get("stride_time_avg_l", 0.0),
                    meta.get("stride_time_avg_r", 0.0),
                    meta.get("stride_time_asymmetry", 0.0),
                    meta.get("step_length_pixel_avg", 0.0),
                ], dtype=np.float32)

            self.valid_sessions.append({
                "folder": folder,
                "issue_text": issue_text,
                "ccc_score": ccc_score,
                "metrics": metrics
            })

        print(f"✅ Dataset ready: {len(self.valid_sessions)} clips loaded.")

    def __len__(self):
        return len(self.valid_sessions)

    def __getitem__(self, idx):
        item = self.valid_sessions[idx]
        session_folder = item["folder"]

        # 1. Load the already-perfected normalized keypoints (Shape: 66)
        kp_path = session_folder / "keypoints_normalized.npy"
        keypoints = np.load(kp_path).astype(np.float32)
        
        # 2. Clean NaNs
        keypoints = clean_keypoints(keypoints)
        
        # 3. FILTER down to just the leg/hip/shoulder features! (Shape: 26)
        keypoints = filter_gait_keypoints(keypoints)

        
        
        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        # 3. Load metrics
        metrics = torch.tensor(item["metrics"], dtype=torch.float32)

        # 4. Targets
        ccc = torch.tensor([item["ccc_score"]], dtype=torch.float32)
        issue_idx = self.class_mapping.get(item["issue_text"], self.class_mapping.get("Unknown / Other", 0))
        issue = torch.tensor(issue_idx, dtype=torch.long)

        return keypoints, metrics, ccc, issue


def pad_collate_fn(batch):
    keypoints_list, metrics_list, ccc_list, issue_list = [], [], [], []
    lengths_list = []

    for kp, met, ccc, iss in batch:
        keypoints_list.append(kp)
        metrics_list.append(met)
        ccc_list.append(ccc)
        issue_list.append(iss)
        lengths_list.append(kp.shape[0])

    padded_keypoints = pad_sequence(
        keypoints_list, batch_first=True, padding_value=0.0
    )

    return (
        padded_keypoints,
        torch.stack(metrics_list),
        torch.stack(ccc_list),
        torch.stack(issue_list),
        torch.tensor(lengths_list, dtype=torch.long)
    )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("\n--- Testing ProGaitDataset ---")

    dataset = ProGaitDataset()
    
    assert len(dataset) > 0, "Dataset empty"

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=pad_collate_fn)

    kp, met, ccc, issue, lengths = next(iter(loader))

    print("Keypoints:", kp.shape)
    print("Metrics:", met.shape)
    print("CCC:", ccc.shape)
    print("Issues:", issue.shape)

    assert not torch.isnan(kp).any(), "NaNs in keypoints"
    assert met.shape[1] == 6, "Wrong metric size"

    print("✔ Dataset OK")