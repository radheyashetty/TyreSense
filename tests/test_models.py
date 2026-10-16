import torch
from src.models import (
    create_gru_model,
    create_rnn_model,
    create_lstm_model,
    count_parameters,
)
from src.train import (
    masked_loss,
    train_one_epoch,
    StintSequenceDataset,
)
from torch.utils.data import DataLoader


def test_models_forward_and_param_counts():
    """Verify forward pass output dimensions and expected parameter counts."""
    gru = create_gru_model()
    rnn = create_rnn_model()
    lstm = create_lstm_model()

    gru_params = count_parameters(gru)
    print(f"GRU parameter count: {gru_params}")
    assert 5000 <= gru_params <= 6000

    batch_size, seq_len = 4, 10
    numeric_x = torch.randn(batch_size, seq_len, 7)
    track_ids = torch.randint(0, 10, (batch_size, seq_len))
    compound_ids = torch.randint(0, 5, (batch_size, seq_len))

    for model in [gru, rnn, lstm]:
        out = model(numeric_x, track_ids, compound_ids)
        assert out.shape == (batch_size, 20)


def test_train_epoch_step():
    """Verify that a single training step runs on CPU without errors."""
    gru = create_gru_model()
    optimizer = torch.optim.Adam(gru.parameters(), lr=1e-3)
    loss_fn = torch.nn.HuberLoss(reduction="none", delta=1.0)

    seqs = {
        "numeric": torch.randn(8, 10, 7).numpy(),
        "track": torch.randint(0, 5, (8, 10)).numpy(),
        "compound": torch.randint(0, 3, (8, 10)).numpy()
    }
    targets = torch.randn(8, 20).numpy()
    targets[0, 5:] = float("nan")

    dataset = StintSequenceDataset(seqs, targets)
    loader = DataLoader(dataset, batch_size=4)

    initial_loss = train_one_epoch(gru, loader, optimizer, loss_fn)
    assert not torch.isnan(torch.tensor(initial_loss))
    assert initial_loss > 0.0
