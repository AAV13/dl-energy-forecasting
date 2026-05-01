import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, d_model, input_size, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        # projects each patch (patch_len * 1 channel) to d_model
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, lookback, 1)
        # squeeze channel dim: (batch, lookback)
        x = x.squeeze(-1)
        # unfold into patches: (batch, num_patches, patch_len)
        x = x.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        x = self.projection(x)  # (batch, num_patches, d_model)
        return self.dropout(x)


class CrossVariableFusion(nn.Module):
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
        # attn_weights: (batch, n_targets, n_targets) -- this is Figure 8
        return out, attn_weights


class PatchTST(nn.Module):
    def __init__(
        self,
        input_size,       # number of input channels (15)
        n_targets,        # number of target variables (4)
        patch_len,        # patch length (24)
        lookback,         # lookback window (168)
        horizon,          # forecast horizon (24)
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.1,
        fusion_heads=2
    ):
        super().__init__()
        self.input_size = input_size
        self.n_targets = n_targets
        self.patch_len = patch_len
        self.horizon = horizon
        self.num_patches = lookback // patch_len  # 168 // 24 = 7

        # one patch embedding per input channel (channel-independent)
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(patch_len, d_model, 1, dropout)
            for _ in range(input_size)
        ])

        # shared transformer encoder across all channels
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # extract temporal attention from encoder for Figure 7
        self.temporal_attn_weights = None

        # pool each channel's patch representations to a single vector
        self.channel_pool = nn.Linear(self.num_patches * d_model, d_model)

        # cross-variable fusion over the 4 target variable representations
        self.fusion = CrossVariableFusion(d_model, num_heads=fusion_heads)

        # output head: project each target's fused representation to horizon steps
        self.output_head = nn.Linear(d_model, horizon)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, lookback, input_size)
        batch_size = x.shape[0]
        channel_reps = []

        for i in range(self.input_size):
            # extract single channel: (batch, lookback, 1)
            ch = x[:, :, i:i+1]
            # patch and embed: (batch, num_patches, d_model)
            ch_patched = self.patch_embeddings[i](ch)
            # encode: (batch, num_patches, d_model)
            ch_encoded = self.encoder(ch_patched)
            # pool patches to single vector: (batch, d_model)
            ch_flat = ch_encoded.reshape(batch_size, -1)
            ch_pooled = self.channel_pool(ch_flat)
            channel_reps.append(ch_pooled)

        # stack all channel representations: (batch, input_size, d_model)
        all_reps = torch.stack(channel_reps, dim=1)

        # extract only target variable representations for fusion
        # targets are: solar(idx2), wind_onshore(idx4), wind_offshore(idx5), price(idx8)
        target_indices = [2, 4, 5, 8]
        target_reps = all_reps[:, target_indices, :]  # (batch, 4, d_model)

        # cross-variable fusion with explicit attention weights
        fused, attn_weights = self.fusion(target_reps)
        # attn_weights: (batch, 4, 4) -- saved for interpretability

        # project each target to forecast horizon
        out = self.output_head(fused)  # (batch, 4, horizon)

        return out, attn_weights
