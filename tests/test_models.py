import torch
from torch.utils.data import DataLoader
from src.models import (
    create_gru_model,
    create_rnn_model,
    create_lstm_model,
    count_parameters,
)
from src.train import (
    train_one_epoch,
    asymmetric_huber_loss,
    StintSequenceDataset,
)


def test_models_forward_and_param_counts():
    """Verify forward pass output dimensions and expected parameter counts."""
    gru = create_gru_model()
    rnn = create_rnn_model()
    lstm = create_lstm_model()

    gru_p, rnn_p, lstm_p = count_parameters(gru), count_parameters(rnn), count_parameters(lstm)
    diff_rnn = abs(rnn_p - gru_p) / gru_p
    diff_lstm = abs(lstm_p - gru_p) / gru_p
    assert diff_rnn <= 0.05, f"RNN params ({rnn_p}) differs from GRU ({gru_p}) by {diff_rnn*100:.1f}%"
    assert diff_lstm <= 0.05, f"LSTM params ({lstm_p}) differs from GRU ({gru_p}) by {diff_lstm*100:.1f}%"

    batch_size, seq_len = 4, 10
    num_x = torch.randn(batch_size, seq_len, 8)
    trk = torch.randint(0, 10, (batch_size, seq_len))
    cmp = torch.randint(0, 5, (batch_size, seq_len))
    for m in [gru, rnn, lstm]:
        assert m(num_x, trk, cmp).shape == (batch_size, 20)


def test_monotone_head_is_non_decreasing():
    """Verify that monotone head produces non-decreasing degradation predictions."""
    gru_mono = create_gru_model(head="monotone")
    num_x = torch.randn(3, 10, 8)
    trk = torch.randint(0, 5, (3, 10))
    cmp = torch.randint(0, 3, (3, 10))
    g_now = torch.tensor([0.2, 0.5, 0.8])

    out = gru_mono(num_x, trk, cmp, g_now=g_now)
    diffs = torch.diff(out, dim=1)
    assert torch.all(diffs >= -1e-6), "Monotone head produced decreasing degradation steps"


def test_asymmetric_huber_loss_weighting():
    """Verify asymmetric Huber loss applies 0.5 weight when actual is slower than predicted."""
    pred = torch.tensor([90.0, 90.0])
    target = torch.tensor([91.0, 89.0])  # lap 1: actual slower (+1s); lap 2: actual faster (-1s)
    loss = asymmetric_huber_loss(pred, target, delta=1.0, positive_weight=0.5)

    base_huber = 0.5 * (1.0 ** 2)
    assert torch.isclose(loss[0], torch.tensor(base_huber * 0.5))
    assert torch.isclose(loss[1], torch.tensor(base_huber * 1.0))


def test_train_epoch_step():
    """Verify that a single training step runs on CPU without errors."""
    gru = create_gru_model()
    opt = torch.optim.Adam(gru.parameters(), lr=1e-3)
    loss_fn = torch.nn.HuberLoss(reduction="none", delta=1.0)
    seqs = {
        "numeric": torch.randn(8, 10, 8).numpy(),
        "track": torch.randint(0, 5, (8, 10)).numpy(),
        "compound": torch.randint(0, 3, (8, 10)).numpy()
    }
    targets = torch.randn(8, 20).numpy()
    targets[0, 5:] = float("nan")
    loader = DataLoader(StintSequenceDataset(seqs, targets), batch_size=4)
    loss = train_one_epoch(gru, loader, opt, loss_fn)
    assert not torch.isnan(torch.tensor(loss)) and loss > 0.0
