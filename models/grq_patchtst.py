"""
GRQ-PatchTST: Gated Residual Quantile PatchTST for European Energy Markets.

Architectural upgrades over base PatchTST:
  1. Explicit Persistence Routing: Extracts T-7 baseline, forces model to learn residuals (deltas).
  2. Gated Temperature Cross-Fusion: Temperature scaling prevents attention collapse; GLU gate dynamically filters noise.
  3. Quantile Output Heads: Predicts 10th, 50th, and 90th percentiles for downside risk quantification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class GatedTemperatureCrossAttention(nn.Module):
    """
    Cross-variable fusion module utilizing Temperature Scaling and a Gated Linear Unit (GLU).
    """
    def __init__(self, d_model: int, num_heads: int = 2, temperature: float = 0.3):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.temperature = temperature

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # GLU projection
        self.glu_proj = nn.Linear(d_model, d_model * 2)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim

        Q = self.q_proj(x).reshape(B, N, H, HD).transpose(1, 2)
        K = self.k_proj(x).reshape(B, N, H, HD).transpose(1, 2)
        V = self.v_proj(x).reshape(B, N, H, HD).transpose(1, 2)

        # Apply Temperature Scaling to Q to sharpen attention weights
        scale = (HD ** 0.5) * self.temperature
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        attn_out = self.out_proj(attn_out)

        # Apply Gated Linear Unit (GLU)
        glu = self.glu_proj(attn_out)
        value, gate = glu.chunk(2, dim=-1)
        gated = value * torch.sigmoid(gate)

        # Residual connection + LayerNorm
        output = self.norm(x + gated)

        # Average attention across heads for visualization
        avg_attn = attn_weights.mean(dim=1)

        return output, avg_attn


class QuantileHead(nn.Module):
    """
    MLP Output Head predicting specific quantiles (e.g., Q10, Q50, Q90).
    """
    def __init__(self, flat_dim: int, horizon: int, n_quantiles: int = 3):
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, horizon * n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.reshape(x.shape[0], self.horizon, self.n_quantiles)


class GRQPatchTST(nn.Module):
    """
    Main Model Architecture: Gated Residual Quantile PatchTST.
    """
    def __init__(
        self,
        input_size: int,
        n_targets: int,
        patch_len: int,
        lookback: int,
        horizon: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        fusion_heads: int = 2,
        fusion_temperature: float = 0.3,
        n_quantiles: int = 3,
        target_feat_indices: Optional[List[Optional[int]]] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_targets = n_targets
        self.horizon = horizon
        self.patch_len = patch_len
        self.num_patches = lookback // patch_len
        self.n_quantiles = n_quantiles
        self.target_feat_indices = target_feat_indices

        patch_dim = patch_len * input_size
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=self.num_patches + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fusion = GatedTemperatureCrossAttention(
            d_model=d_model,
            num_heads=fusion_heads,
            temperature=fusion_temperature,
        )

        flat_dim = self.num_patches * d_model
        self.output_heads = nn.ModuleList(
            [QuantileHead(flat_dim, horizon, n_quantiles) for _ in range(n_targets)]
        )

        self.dropout = nn.Dropout(dropout)

    def _persistence_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the first 'horizon' hours from the lookback to act as the T-7 day baseline.
        If a target index is None (e.g., Price), it returns zeros so the model predicts the absolute value.
        """
        B = x.shape[0]
        parts = []
        for i in range(self.n_targets):
            if (self.target_feat_indices is not None 
                and i < len(self.target_feat_indices) 
                and self.target_feat_indices[i] is not None):
                feat_idx = self.target_feat_indices[i]
                base = x[:, : self.horizon, feat_idx]
            else:
                base = torch.zeros(B, self.horizon, device=x.device)
            parts.append(base)
        return torch.stack(parts, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape

        # 1. Extract Persistence baseline
        baseline = self._persistence_baseline(x)

        # 2. Patching and Embedding
        x_p = x.reshape(B, self.num_patches, self.patch_len, C)
        patches = x_p.reshape(B, self.num_patches, -1)
        emb = self.patch_embedding(patches)
        emb = self.pos_encoding(emb)

        # 3. Encoding
        enc = self.encoder(emb)
        flat = enc.reshape(B, -1)

        # 4. Target representations & Fusion
        target_reps = flat[:, : self.n_targets * 128].reshape(B, self.n_targets, 128)
        fused, attn_weights = self.fusion(target_reps)

        fused_flat = fused.reshape(B, -1)
        flat = flat.clone()
        flat[:, : fused_flat.shape[1]] += fused_flat

        # 5. Output Heads (Predict Deltas)
        deltas = [head(flat) for head in self.output_heads]
        delta_out = torch.stack(deltas, dim=2)

        # 6. Residual Addition: Final Output = Baseline + Deltas
        # Expand baseline to broadcast across the 3 quantiles
        baseline_exp = baseline.unsqueeze(-1)
        output = baseline_exp + delta_out

        return output, attn_weights

if __name__ == "__main__":
    # Sanity Check Block
    model = GRQPatchTST(
        input_size=15, n_targets=3, patch_len=24, lookback=168,
        horizon=24, target_feat_indices=[2, 4, None]
    )
    dummy_x = torch.randn(4, 168, 15)
    out, attn = model(dummy_x)
    print("Testing GRQ-PatchTST instantiation and forward pass...")
    print(f"Input shape:     {dummy_x.shape}")
    print(f"Output shape:    {out.shape} -> Expected: (4, 24, 3, 3) [Batch, Horizon, Targets, Quantiles]")
    print(f"Attention shape: {attn.shape} -> Expected: (4, 3, 3)")
    assert out.shape == (4, 24, 3, 3), "Output shape mismatch!"
    print("All checks passed.")