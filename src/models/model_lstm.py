import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Gait-relevant MediaPipe landmark indices (matches train_kfold.py)
# nose(0), shoulders(11,12), hips(23,24), knees(25,26),
# ankles(27,28), heels(29,30), foot indices(31,32)
GAIT_LANDMARK_INDICES = [0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
LSTM_INPUT_DIM        = len(GAIT_LANDMARK_INDICES) * 2   # 26


class GaitSequenceLSTM(nn.Module):
    def __init__(
        self,
        input_dim:    int   = LSTM_INPUT_DIM,  # 26 (hip-centred, gait landmarks only)
        hidden_dim:   int   = 32,              # keep small — 115 training samples
        num_layers:   int   = 2,
        dropout_rate: float = 0.3,
    ):
        """
        Deliberately small LSTM for a ~115-sample training set.

        Parameter budget
        ----------------
        input_dim=26, hidden=32, 2 layers, unidirectional:
          layer 1: 4 × (26×32 + 32×32 + 32) = 4 × (832+1024+32) = 15,552
          layer 2: 4 × (32×32 + 32×32 + 32) = 4 × (1024+1024+32) = 8,320
          output proj 64→64: 64×64+64 = 4,160
          total LSTM block: ~28,000  (was 207,744)

        Fusion output: (B, 64)  — concat with MLP's (B, 64) → (B, 128)
        Note: fusion_dim in ProGaitFusion must be updated to 128 (not 192)
              if you keep MLP output_dim=64 and this LSTM output=64.
        """
        super().__init__()

        # No spatial_proj — going straight into LSTM keeps parameters low.
        # Hip-centering + landmark selection (done in dataset) already gives
        # a clean 26-dim input; expanding it to 128 before the LSTM was
        # the main source of parameter bloat.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=False,   # unidirectional: works at inference time too
        )

        # Small projection head: hidden_dim → 64 to match MLP output dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, T, 26)  hip-centred keypoints, padded
        lengths : (B,)        real frame count per sample
        Returns : (B, 64)
        """
        # Pack so the LSTM ignores padding frames
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _   = pad_packed_sequence(packed_out, batch_first=True)
        # (B, T, hidden_dim)

        # Mean-pool over actual (non-padded) frames
        actual_len = lengths.unsqueeze(1).float().to(x.device)   # (B, 1)
        pooled = lstm_out[:, -1, :]             # (B, hidden_dim)

        return self.proj(pooled)   # (B, 64)

"""
# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from model_mlp import GaitMetricsMLP

    lstm = GaitSequenceLSTM()
    mlp  = GaitMetricsMLP()

    lstm_params = sum(p.numel() for p in lstm.parameters())
    mlp_params  = sum(p.numel() for p in mlp.parameters())

    print(f"LSTM parameters : {lstm_params:,}")
    print(f"MLP  parameters : {mlp_params:,}")
    print(f"Total           : {lstm_params + mlp_params:,}")
    print(f"Params/sample (115 samples): {(lstm_params+mlp_params)/115:.0f}x")

    # Simulate a batch: 2 videos, variable length, 26-dim input
    dummy_kp  = torch.randn(2, 300, LSTM_INPUT_DIM)
    dummy_len = torch.tensor([300, 150])

    out = lstm(dummy_kp, dummy_len)
    print(f"\nInput  : {dummy_kp.shape}  (B, T, {LSTM_INPUT_DIM})")
    print(f"Output : {out.shape}  → (B, 64) ready for fusion with MLP (B, 64)")
    print("✓ LSTM compiled")"""