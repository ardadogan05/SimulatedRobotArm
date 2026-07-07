from pathlib import Path

import numpy as np
import torch

from arms.planar_2link import forward_kinematics
from models.neural_ik_2link import NeuralIK2Link
from training.torch_utils import get_torch_device


def main():
    #path for the data used for testing
    dataset_path = Path("data/processed/ik_2link_dataset.npz")
    model_path = Path("models/saved/neural_ik_2link.pt") #model that is used

    #Loading the data
    data = np.load(dataset_path)

    X_test = data["X_test"]
    Y_test = data["Y_test"]

    #Uses gpu if availabe, cpu otherwise
    device = get_torch_device()

    print("Using device:", device)

    #loads model and puts in eval mode.
    model = NeuralIK2Link().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    #Converts test data to torch.tensor for use in model
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    #Makes sure the model doesnt track gradient as this a test
    with torch.no_grad():
        Y_pred_tensor = model(X_test_tensor)

    #Converts the tensor to regular numbers for further use
    Y_pred = Y_pred_tensor.cpu().numpy()

    angle_errors = np.linalg.norm(Y_pred - Y_test, axis=1)

    #Creates an np array with end effector errors.
    #Allows us to use functions like mean, median max etc.
    ee_errors = []

    for target_xy, pred_theta in zip(X_test, Y_pred):
        theta1_pred, theta2_pred = pred_theta

        _, _, end_effector = forward_kinematics(theta1_pred, theta2_pred)

        ee_error = np.linalg.norm(end_effector - target_xy)
        ee_errors.append(ee_error)

    ee_errors = np.array(ee_errors)

    print("Neural IK test-set evaluation")
    print("-----------------------------")
    print(f"Mean angle error:   {np.mean(angle_errors):.6f} rad")
    print(f"Median angle error: {np.median(angle_errors):.6f} rad")
    print(f"Max angle error:    {np.max(angle_errors):.6f} rad")
    print()
    print(f"Mean EE error:      {np.mean(ee_errors):.6f}")
    print(f"Median EE error:    {np.median(ee_errors):.6f}")
    print(f"Max EE error:       {np.max(ee_errors):.6f}")


if __name__ == "__main__":
    main()
