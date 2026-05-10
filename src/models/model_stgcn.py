import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# ADJACENCY MATRIX
# ============================================================

def build_adjacency_matrix(num_joints, connections):
    """
    Creates normalized adjacency matrix for ST-GCN.

    Args:
        num_joints: total number of joints
        connections: list of tuples (joint_a, joint_b)

    Returns:
        torch.Tensor shape: (V, V)
    """

    A = np.zeros((num_joints, num_joints), dtype=np.float32)

    # Undirected edges
    for i, j in connections:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # Self-connections
    A += np.eye(num_joints, dtype=np.float32)

    # Degree normalization
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0

    D_inv_sqrt = np.diag(D_inv_sqrt)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.tensor(A_norm, dtype=torch.float32)


# ============================================================
# ST-GCN BLOCK
# ============================================================

class STGCNBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        temporal_kernel_size=9,
        dropout=0.2
    ):
        super().__init__()

        padding = ((temporal_kernel_size - 1) // 2, 0)

        # Spatial graph convolution
        self.spatial_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel_size, 1),
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

        # Residual connection
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A, edge_importance):

        residual = self.residual(x)

        # Spatial graph convolution
        x = self.spatial_conv(x)

        # Graph propagation
        x = torch.einsum(
            'nctv,vw->nctw',
            x,
            A * edge_importance
        )

        # Temporal convolution
        x = self.temporal_conv(x)

        x = x + residual

        return self.relu(x)


# ============================================================
# MAIN MODEL
# ============================================================

class GaitSTGCN(nn.Module):

    def __init__(
        self,
        num_joints=13,
        num_classes=9,
        dropout=0.3
    ):
        super().__init__()

        # ====================================================
        # GRAPH DEFINITION (MUST MATCH DATASET JOINT ORDER)
        # ====================================================
        #
        # Your dataset does NOT feed MediaPipe indices (11/12/23/24/...) into the model.
        # `ProGaitDataset.filter_gait_keypoints()` selects 13 landmarks by index from
        # `GAIT_LANDMARK_INDICES` and packs them densely into a (T, 13*2) feature vector.
        #
        # So the ST-GCN graph MUST be defined in that compact index space: [0..num_joints-1].
        # If you build edges with MediaPipe IDs, you silently build the wrong graph for the
        # data you're actually feeding, and training often gets stuck near chance.
        #
        # We'll import the landmark list and build a "MediaPipe->compact" mapping.

        try:
            from src.models.model_lstm import GAIT_LANDMARK_INDICES  # re-used by dataset
        except Exception as e:
            raise ImportError(
                "Could not import GAIT_LANDMARK_INDICES from src.models.model_lstm. "
                "The ST-GCN graph must match the dataset joint ordering."
            ) from e

        if num_joints != len(GAIT_LANDMARK_INDICES):
            raise ValueError(
                f"num_joints={num_joints} does not match len(GAIT_LANDMARK_INDICES)={len(GAIT_LANDMARK_INDICES)}. "
                "Make them consistent (dataset outputs V*2 features)."
            )

        self.gait_landmark_indices = list(GAIT_LANDMARK_INDICES)
        mp_to_compact = {mp_idx: i for i, mp_idx in enumerate(self.gait_landmark_indices)}

        # --- Leg-focused connections in *MediaPipe landmark index* space ---
        # The gait landmark set is: [0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        # We'll focus on pelvis + legs (optionally keep shoulders as an anchor).
        raw_connections_mp = [
            # shoulders -> hips (optional anchor)
            (11, 23),
            (12, 24),

            # hip -> knee -> ankle
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28),

            # ankle -> heel -> foot_index
            (27, 29),
            (29, 31),
            (28, 30),
            (30, 32),

            # left/right pelvis connection
            (23, 24),
        ]

        # Keep only those edges whose endpoints exist in GAIT_LANDMARK_INDICES.
        # (If an edge gets dropped, it's because that joint isn't in the dataset features.)
        connections = []
        dropped = 0
        for a, b in raw_connections_mp:
            if a in mp_to_compact and b in mp_to_compact:
                connections.append((mp_to_compact[a], mp_to_compact[b]))
            else:
                dropped += 1

        if len(connections) == 0:
            raise ValueError(
                "All requested graph edges were dropped because the connected joints are not present "
                "in GAIT_LANDMARK_INDICES. Update edges or update the selected landmarks."
            )

        self.dropped_edges = dropped

        # ====================================================
        # ADJACENCY MATRIX
        # ====================================================

        A = build_adjacency_matrix(
            num_joints=num_joints,
            connections=connections
        )

        self.register_buffer("A", A)

        # Learnable edge importance
        self.edge_importance = nn.Parameter(
            torch.ones_like(A)
        )

        # ====================================================
        # ST-GCN STACK
        # ====================================================

        self.block1 = STGCNBlock(
            in_channels=4,   # x,y,vx,vy
            out_channels=64,
            dropout=dropout
        )

        self.block2 = STGCNBlock(
            in_channels=64,
            out_channels=128,
            dropout=dropout
        )

        self.block3 = STGCNBlock(
            in_channels=128,
            out_channels=256,
            dropout=dropout
        )

        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

        # Small but useful: stable init for classifier.
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x, lengths=None):
        """
        Input:
            x shape:
            (B, T, V*2)

            Example:
            B = batch
            T = frames
            V = joints

        Output:
            logits: (B, num_classes)
        """

        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape (B,T,F) but got {tuple(x.shape)}")

        B, T, F = x.shape
        V = self.A.shape[0]

        if F != V * 2:
            raise ValueError(
                f"Expected feature dim F == V*2 == {V*2} but got F={F}. "
                "This usually means dataset filtering (GAIT_LANDMARK_INDICES) and model num_joints disagree."
            )

        # --------------------------------------------
        # reshape
        # --------------------------------------------

        x = x.view(B, T, V, 2)

        # --------------------------------------------
        # velocity stream
        # --------------------------------------------

        velocity = torch.zeros_like(x)
        velocity[:, 1:] = x[:, 1:] - x[:, :-1]

        # concatenate:
        # x,y,vx,vy
        x = torch.cat([x, velocity], dim=-1)

        # shape:
        # (B,T,V,4)

        # ST-GCN expects:
        # (B,C,T,V)

        x = x.permute(0, 3, 1, 2)  # (B,C,T,V)

        # Optional: mask padded frames so they don't dominate temporal stats.
        # (Your dataset appears fixed-length 150 frames, but pad_collate_fn still exists.)
        if lengths is not None:
            # lengths: (B,)
            if lengths.dim() != 1 or lengths.numel() != B:
                raise ValueError("lengths must have shape (B,)")
            mask = (
                torch.arange(T, device=x.device)[None, :] < lengths[:, None]
            ).float()  # (B,T)
            x = x * mask[:, None, :, None]

        # --------------------------------------------
        # ST-GCN blocks
        # --------------------------------------------

        x = self.block1(
            x,
            self.A,
            self.edge_importance
        )

        x = self.block2(
            x,
            self.A,
            self.edge_importance
        )

        x = self.block3(
            x,
            self.A,
            self.edge_importance
        )

        # --------------------------------------------
        # Global pooling
        # --------------------------------------------

        x = x.mean(dim=-1)   # average joints => (B,C,T)

        if lengths is None:
            x = x.mean(dim=-1)  # average time
        else:
            # masked mean over time
            denom = lengths.clamp(min=1).float().to(x.device).unsqueeze(1)  # (B,1)
            x = x.sum(dim=-1) / denom  # (B,C)

        # shape:
        # (B,256)

        # --------------------------------------------
        # classifier
        # --------------------------------------------

        logits = self.classifier(x)

        return logits