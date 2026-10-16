import torch
import torch.nn as nn


def count_parameters(model):
    """Count the total number of trainable parameters in a PyTorch model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class RecurrentForecaster(nn.Module):
    """Base recurrent model forecasting fuel-corrected residuals for 20 laps."""

    def __init__(self, rnn_type="gru", num_tracks=30, num_compounds=10):
        """Build embedding, recurrent cell, and 2-layer MLP head."""
        super().__init__()
        self.track_emb = nn.Embedding(num_tracks, 4)
        self.compound_emb = nn.Embedding(num_compounds, 3)
        self.input_drop = nn.Dropout(0.1)

        input_dim = 7 + 4 + 3  # 7 numeric + 4 track emb + 3 compound emb = 14
        if rnn_type == "rnn":
            self.rnn = nn.RNN(input_dim, 32, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_dim, 32, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, 32, batch_first=True)

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 20)
        )

    def forward(self, numeric_x, track_ids, compound_ids):
        """Pass input sequence through embeddings, recurrent layer, and MLP head."""
        trk_e = self.track_emb(track_ids)
        cmp_e = self.compound_emb(compound_ids)
        x = torch.cat([numeric_x, trk_e, cmp_e], dim=-1)
        x = self.input_drop(x)

        out, _ = self.rnn(x)
        last_step = out[:, -1, :]
        return self.head(last_step)


def create_gru_model(num_tracks=30, num_compounds=10):
    """Instantiate the primary GRU forecaster model D."""
    return RecurrentForecaster(rnn_type="gru", num_tracks=num_tracks, num_compounds=num_compounds)


def create_rnn_model(num_tracks=30, num_compounds=10):
    """Instantiate the vanilla RNN forecaster for EXP-7."""
    return RecurrentForecaster(rnn_type="rnn", num_tracks=num_tracks, num_compounds=num_compounds)


def create_lstm_model(num_tracks=30, num_compounds=10):
    """Instantiate the LSTM forecaster for EXP-7."""
    return RecurrentForecaster(rnn_type="lstm", num_tracks=num_tracks, num_compounds=num_compounds)
