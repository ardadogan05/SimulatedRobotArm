import torch

from models.neural_ik_3link import NeuralIK3Link, forward_kinematics_3link_torch
from training.train_utils import train_neural_ik


def end_effector_loss(predicted_delta_theta, model_input):
    target = model_input[:, 0:2]
    current_theta = model_input[:, 2:5]

    theta_after_update = current_theta + predicted_delta_theta
    predicted_end_effector = forward_kinematics_3link_torch(theta_after_update)

    return torch.mean((predicted_end_effector - target) ** 2)


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
        #The upgraded model plateaued before 50 epochs during the first full run
        n_epochs=50,
        batch_size=512,
        learning_rate=1e-3,
        task_loss_fn=end_effector_loss,
        task_loss_weight=0.1,
    )


if __name__ == "__main__":
    main()
