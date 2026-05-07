import torch
import torch.nn as nn

class BiLSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.1, horizon=24, n_targets=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # bidirectional doubles hidden size
        self.output_head = nn.Linear(hidden_size * 2, n_targets * horizon)
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x):
        # x: (batch, lookback, input_size)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])          # take last timestep
        out = self.output_head(out)                # (batch, n_targets * horizon)
        return out.view(-1, self.n_targets, self.horizon)