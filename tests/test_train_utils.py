import numpy as np
import pytest
import torch
from torch import nn

import training.train_utils as train_utils


def make_tiny_model():
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )


def test_train_neural_ik_saves_model_and_losses(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train_utils,
        "get_torch_device",
        lambda: torch.device("cpu"),
    )

    X_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )
    Y_train = 2.0 * X_train
    X_val = np.array([[0.25, 0.75], [0.75, 0.25]])
    Y_val = 2.0 * X_val

    dataset_path = tmp_path / "dataset.npz"
    model_path = tmp_path / "models" / "model.pt"
    losses_path = tmp_path / "results" / "losses.npz"

    np.savez(
        dataset_path,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
    )

    torch.manual_seed(0)
    model = make_tiny_model()

    #This is the full-dataset MSE that the weighted mini-batch average should match
    with torch.no_grad():
        expected_train_loss = nn.MSELoss()(
            model(torch.tensor(X_train, dtype=torch.float32)),
            torch.tensor(Y_train, dtype=torch.float32),
        ).item()
        expected_val_loss = nn.MSELoss()(
            model(torch.tensor(X_val, dtype=torch.float32)),
            torch.tensor(Y_val, dtype=torch.float32),
        ).item()

    train_losses, val_losses = train_utils.train_neural_ik(
        model,
        dataset_path=dataset_path,
        model_path=model_path,
        losses_path=losses_path,
        n_epochs=2,
        batch_size=2,
        learning_rate=0.0,
    )

    assert len(train_losses) == 2
    assert len(val_losses) == 2
    assert np.isfinite(train_losses).all()
    assert np.isfinite(val_losses).all()

    #With a zero learning rate the model stays unchanged, so both epochs match
    assert train_losses == pytest.approx([expected_train_loss] * 2)
    assert val_losses == pytest.approx([expected_val_loss] * 2)

    assert model_path.exists()
    assert losses_path.exists()

    saved_model = make_tiny_model()
    saved_model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )

    with np.load(losses_path) as losses:
        assert set(losses.files) == {"train_losses", "val_losses"}
        assert losses["train_losses"].shape == (2,)
        assert losses["val_losses"].shape == (2,)
