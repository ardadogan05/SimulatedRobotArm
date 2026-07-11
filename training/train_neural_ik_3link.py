from models.neural_ik_3link import NeuralIK3Link
from training.train_utils import train_neural_ik


def main():
    #The 3-link NN learns a movement update, not final angles
    #[target x, target y, current theta] -> [delta theta1, delta theta2, delta theta3]
    model = NeuralIK3Link()

    #The shared function contains the PyTorch training loop used by both NNs
    train_neural_ik(
        model,
        dataset_path="data/processed/ik_3link_dataset.npz",
        model_path="models/saved/neural_ik_3link.pt",
        losses_path="results/phase8/neural_ik_3link_losses.npz",
        n_epochs=100,
        batch_size=128,
        learning_rate=1e-3,
    )


if __name__ == "__main__":
    main()
