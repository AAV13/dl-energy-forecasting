import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe: (max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, num_patches, d_model)
        return x + self.pe[:, :x.size(1), :]


class CrossVariableFusion(nn.Module):
    """Our extension — cross-attention over the 3 target variable representations."""
    def __init__(self, d_model, num_heads=2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, n_targets, d_model)
        attn_out, attn_weights = self.cross_attn(x, x, x)
        out = self.norm(x + attn_out)
        # attn_weights: (batch, n_targets, n_targets) — Figure 8
        return out, attn_weights


class PatchTST(nn.Module):
    def __init__(
        self,
        input_size,       # 12 input channels
        n_targets,        # 3 target variables
        patch_len,        # 24 hours per patch
        lookback,         # 168 hours
        horizon,          # 24 hours
        d_model=128,
        n_heads=8,
        n_layers=3,
        d_ff=512,
        dropout=0.1,
        use_fusion=False, # PatchTST+ when True
        fusion_heads=2
    ):
        super().__init__()
        self.use_fusion = use_fusion
        self.n_targets = n_targets
        self.horizon = horizon
        self.num_patches = lookback // patch_len  # 168 // 24 = 7

        # ── PATCH EMBEDDING ──────────────────────────────────────────
        # Joint tokenization: all channels flattened into each patch
        # Each patch: patch_len × input_size = 24 × 12 = 288 dims
        patch_dim = patch_len * input_size
        self.patch_len = patch_len
        self.input_size = input_size

        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # ── POSITIONAL ENCODING ───────────────────────────────────────
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=self.num_patches + 10)

        # ── TRANSFORMER ENCODER ───────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True   # Pre-Norm as shown in slides
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── OPTIONAL FUSION (PatchTST+) ───────────────────────────────
        if use_fusion:
            self.fusion = CrossVariableFusion(d_model, num_heads=fusion_heads)

        # ── OUTPUT HEADS — one per target ────────────────────────────
        # Flattened encoder output: num_patches × d_model = 7 × 128 = 896
        flat_dim = self.num_patches * d_model

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat_dim, 256),
                nn.GELU(),
                nn.Linear(256, horizon)
            )
            for _ in range(n_targets)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, lookback, input_size) = (B, 168, 12)
        batch_size = x.shape[0]

        # ── PATCH + EMBED ─────────────────────────────────────────────
        # Unfold into patches: (B, num_patches, patch_len, input_size)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        # Flatten channels into patch: (B, num_patches, patch_len * input_size)
        patches = patches.reshape(batch_size, self.num_patches, -1)
        # Project to d_model: (B, num_patches, d_model)
        embedded = self.patch_embedding(patches)

        # ── POSITIONAL ENCODING ───────────────────────────────────────
        embedded = self.pos_encoding(embedded)

        # ── TRANSFORMER ENCODER ───────────────────────────────────────
        encoded = self.encoder(embedded)   # (B, num_patches, d_model)

        # ── FLATTEN ───────────────────────────────────────────────────
        flat = encoded.reshape(batch_size, -1)  # (B, num_patches * d_model) = (B, 896)

        # ── OPTIONAL FUSION (PatchTST+) ───────────────────────────────
        attn_weights = None
        if self.use_fusion:
            # Project flat to per-target representations for fusion
            # Reshape to (B, n_targets, d_model) using first n_targets * d_model dims
            target_reps = flat[:, :self.n_targets * 128].reshape(batch_size, self.n_targets, 128)
            fused, attn_weights = self.fusion(target_reps)
            # Inject fused info back — add to flat representation
            flat = flat + fused.reshape(batch_size, -1)[:, :flat.shape[1]]

        # ── OUTPUT HEADS ──────────────────────────────────────────────
        outputs = []
        for head in self.output_heads:
            outputs.append(head(flat))  # each: (B, horizon)

        # Stack: (B, n_targets, horizon)
        out = torch.stack(outputs, dim=1)

        return out, attn_weights