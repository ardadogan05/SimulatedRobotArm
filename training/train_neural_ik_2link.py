from models.neural_ik_2link import NeuralIK2Link
from training.train_utils import train_neural_ik


def main():
    #The 2-link NN learns final angles: [x, y] -> [theta1, theta2]
    model = NeuralIK2Link()

    #The shared function contains the PyTorch training loop used by both NNs
    train_neural_ik(
        model,
        dataset_path="data/processed/ik_2link_dataset.npz",
        model_path="models/saved/neural_ik_2link.pt",
        losses_path="results/phase4/neural_ik_2link_losses.npz",
        n_epochs=100,
        batch_size=128,
        learning_rate=1e-3,
    )


if __name__ == "__main__":
    main()
