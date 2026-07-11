import numpy as np
import torch

from arms.planar_3link import forward_kinematics_3link
from solvers.utils import wrap_to_pi


def neural_policy_step_3link(model, target, theta):
    target = np.array(target, dtype=float)
    theta = np.array(theta, dtype=float)

    #Input: [target x, target y, current theta1, current theta2, current theta3]
    model_input = np.concatenate([target, theta])

    #Uses the same device as the model, either CPU or GPU
    device = next(model.parameters()).device
    model_input = torch.tensor(
        model_input,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        predicted_delta_theta = model(model_input)[0]

    return predicted_delta_theta.cpu().numpy()


def neural_solver_3link(
    model,
    target,
    initial_theta=None,
    tolerance=0.05,
    max_iterations=100,
    max_step_norm=0.5,
):
    target = np.array(target, dtype=float)

    if initial_theta is None:
        initial_theta = [0.1, 0.1, 0.1]

    theta = np.array(initial_theta, dtype=float)

    #Stores the complete rollout so it can be inspected and plotted later
    theta_history = [theta.copy()]
    error_history = []

    for iteration in range(max_iterations + 1):
        current_position = forward_kinematics_3link(
            theta[0],
            theta[1],
            theta[2],
        )[3]

        error_norm = np.linalg.norm(target - current_position)
        error_history.append(error_norm)

        if error_norm < tolerance:
            return {
                "theta": theta,
                "success": True,
                "final_error": error_norm,
                "iterations": iteration,
                "theta_history": np.array(theta_history),
                "error_history": np.array(error_history),
            }

        if iteration == max_iterations:
            break

        predicted_delta_theta = neural_policy_step_3link(
            model,
            target,
            theta,
        )

        #Uses the same maximum movement size as the training dataset
        step_norm = np.linalg.norm(predicted_delta_theta)
        if max_step_norm is not None and step_norm > max_step_norm:
            predicted_delta_theta = (
                predicted_delta_theta * max_step_norm / step_norm
            )

        #The training targets already contain the DLS learning rate.
        #Therefore the prediction is applied directly without scaling it again.
        theta = theta + predicted_delta_theta
        theta = wrap_to_pi(theta)
        theta_history.append(theta.copy())

    return {
        "theta": theta,
        "success": False,
        "final_error": error_norm,
        "iterations": max_iterations,
        "theta_history": np.array(theta_history),
        "error_history": np.array(error_history),
    }
