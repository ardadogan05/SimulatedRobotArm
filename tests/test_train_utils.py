from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from torch import nn

from training.train_utils import train_neural_ik


def test_train_neural_ik():
    X_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    Y_train = 2.0 * X_train

    X_val = np.array([[0.25, 0.75], [0.75, 0.25]])
    Y_val = 2.0 * X_val

    #Creates a temporary folder inside tests and removes it when the test finishes
    tests_dir = Path(__file__).parent
    with TemporaryDirectory(dir=tests_dir) as temporary_folder:
        temporary_folder = Path(temporary_folder)

        dataset_path = temporary_folder / "dataset.npz"
        model_path = temporary_folder / "model.pt"
        losses_path = temporary_folder / "losses.npz"

        np.savez(
            dataset_path,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
        )

        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

        #Only one epoch is needed to check that the shared training function runs
        train_losses, val_losses = train_neural_ik(
            model,
            dataset_path=dataset_path,
            model_path=model_path,
            losses_path=losses_path,
            n_epochs=1,
            batch_size=2,
        )

        assert len(train_losses) == 1
        assert len(val_losses) == 1
        assert model_path.exists()
        assert losses_path.exists()
