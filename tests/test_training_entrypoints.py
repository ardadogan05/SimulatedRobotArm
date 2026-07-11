from models.neural_ik_2link import NeuralIK2Link
from models.neural_ik_3link import NeuralIK3Link
import training.train_neural_ik_2link as train_2link
import training.train_neural_ik_3link as train_3link


def capture_training_call(module, monkeypatch):
    call = {}

    def fake_train(model, **kwargs):
        call["model"] = model
        call.update(kwargs)

    monkeypatch.setattr(module, "train_neural_ik", fake_train)
    module.main()
    return call


def test_2link_training_entrypoint_uses_expected_configuration(monkeypatch):
    call = capture_training_call(train_2link, monkeypatch)

    assert isinstance(call["model"], NeuralIK2Link)
    assert call["dataset_path"] == "data/processed/ik_2link_dataset.npz"
    assert call["model_path"] == "models/saved/neural_ik_2link.pt"
    assert call["losses_path"] == "results/phase4/neural_ik_2link_losses.npz"
    assert call["n_epochs"] == 100
    assert call["batch_size"] == 128
    assert call["learning_rate"] == 1e-3


def test_3link_training_entrypoint_uses_expected_configuration(monkeypatch):
    call = capture_training_call(train_3link, monkeypatch)

    assert isinstance(call["model"], NeuralIK3Link)
    assert call["dataset_path"] == "data/processed/ik_3link_dataset.npz"
    assert call["model_path"] == "models/saved/neural_ik_3link.pt"
    assert call["losses_path"] == "results/phase8/neural_ik_3link_losses.npz"
    assert call["n_epochs"] == 100
    assert call["batch_size"] == 128
    assert call["learning_rate"] == 1e-3
