import torch
import torch.nn as nn


def count_parameters(model):
    """Count the total number of trainable parameters in a PyTorch model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class MonotoneHead(nn.Module):
    """Head enforcing monotonically non-decreasing degradation predictions."""

    def __init__(self, hidden_dim, horizon=20):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(hidden_dim, 16), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(16, horizon)
        )

    def forward(self, h, g_now=None):
        steps = torch.relu(self.fc(h))
        cum_steps = torch.cumsum(steps, dim=1)
        if g_now is not None:
            if g_now.dim() == 1:
                g_now = g_now.unsqueeze(1)
            return g_now + cum_steps
        return cum_steps


class RecurrentForecaster(nn.Module):
    """Recurrent model forecasting fuel-corrected residuals for 20 laps."""

    def __init__(self, rnn_type="gru", hidden_dim=32, num_numeric=8, head_type="free", num_tracks=50, num_compounds=10):
        super().__init__()
        self.track_emb = nn.Embedding(num_tracks, 4)
        self.compound_emb = nn.Embedding(num_compounds, 3)
        self.input_drop = nn.Dropout(0.1)
        self.head_type = head_type

        input_dim = num_numeric + 4 + 3
        if rnn_type == "rnn":
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

        if head_type == "monotone":
            self.head = MonotoneHead(hidden_dim, 20)
        else:
            self.head = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(hidden_dim, 16), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(16, 20)
            )

    def forward(self, numeric_x, track_ids, compound_ids, g_now=None):
        trk_e = self.track_emb(track_ids)
        cmp_e = self.compound_emb(compound_ids)
        x = torch.cat([numeric_x, trk_e, cmp_e], dim=-1)
        out, _ = self.rnn(self.input_drop(x))
        last_step = out[:, -1, :]
        if self.head_type == "monotone":
            return self.head(last_step, g_now=g_now)
        return self.head(last_step)


def create_gru_model(hidden=32, num_numeric=8, head="free"):
    """Instantiate primary GRU forecaster model D with matched budget."""
    return RecurrentForecaster("gru", hidden_dim=hidden, num_numeric=num_numeric, head_type=head)


def create_rnn_model(hidden=58, num_numeric=8, head="free"):
    """Instantiate vanilla RNN forecaster with matched 5% parameter budget."""
    return RecurrentForecaster("rnn", hidden_dim=hidden, num_numeric=num_numeric, head_type=head)


def create_lstm_model(hidden=27, num_numeric=8, head="free"):
    """Instantiate LSTM forecaster with matched 5% parameter budget."""
    return RecurrentForecaster("lstm", hidden_dim=hidden, num_numeric=num_numeric, head_type=head)
