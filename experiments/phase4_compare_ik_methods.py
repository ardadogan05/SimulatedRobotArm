from pathlib import Path

import numpy as np
import torch

from arms.planar_2link import forward_kinematics
from models.neural_ik_2link import NeuralIK2Link
from solvers.analytical_2link import analytical_ik_2link
from solvers.jacobian_ik_2link import numerical_solver_2link
from training.torch_utils import get_torch_device


def choose_positive_theta2_solution(solutions):
    """
    analytical_ik_2link returns:
        [(theta1_a, theta2_a), (theta1_b, theta2_b)]

    Our neural dataset was generated with theta2 in [0, pi],
    so we choose the branch where theta2 is positive.
    """
    if solutions is None:
        return None

    for theta in solutions:
        if theta[1] >= 0:
            return np.array(theta)

    return np.array(solutions[0])


def end_effector_error(theta, target_xy):
    theta1, theta2 = theta

    _, _, end_effector = forward_kinematics(theta1, theta2)

    return np.linalg.norm(end_effector - target_xy)


def summarize_errors(name, errors):
    errors = np.array(errors)

    return {
        "method": name,
        "mean": np.mean(errors),
        "median": np.median(errors),
        "max": np.max(errors),
    }


def main():
    dataset_path = Path("data/processed/ik_2link_dataset.npz") #path for the data used for testing
    model_path = Path("models/saved/neural_ik_2link.pt") #model that is used


    data = np.load(dataset_path)

    X_test = data["X_test"]

    #Uses gpu if availabe, cpu otherwise
    device = get_torch_device()

    #loads model and puts in eval mode.
    model = NeuralIK2Link().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    #Converts test data to torch.tensor for use in model
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    #Makes sure the model doesnt track gradient as this a test
    with torch.no_grad():
        Y_pred = model(X_test_tensor).cpu().numpy()

    #List to keep track of errors
    analytical_errors = []
    numerical_errors = []
    neural_errors = []

    # Start with 1000 samples because numerical IK is slower than analytical/neural.
    num_samples = 1000

    for i in range(num_samples):
        target_xy = X_test[i]
        neural_theta = Y_pred[i]

        # Analytical IK
        analytical_solutions = analytical_ik_2link(target_xy)
        theta_analytical = choose_positive_theta2_solution(analytical_solutions)

        if theta_analytical is not None:
            analytical_errors.append(
                end_effector_error(theta_analytical, target_xy)
            )

        
        # Numerical IK
        initial_theta = np.array([0.0, 0.0])

        numerical_result = numerical_solver_2link(
            target_xy,
            initial_theta=initial_theta,
        )

        theta_numerical = numerical_result["theta"]

        numerical_errors.append(
            end_effector_error(theta_numerical, target_xy)
        )

        # Neural IK
        neural_errors.append(
            end_effector_error(neural_theta, target_xy)
        )

    summaries = [
        summarize_errors("Analytical IK", analytical_errors),
        summarize_errors("Numerical IK", numerical_errors),
        summarize_errors("Neural IK", neural_errors),
    ]

    #For printing a nice table
    print()
    print("Phase 4 IK method comparison")
    print("----------------------------")
    print(
        f"{'Method':<18} "
        f"{'Mean EE error':>16} "
        f"{'Median EE error':>18} "
        f"{'Max EE error':>14}"
    )

    for s in summaries:
        print(
            f"{s['method']:<18} "
            f"{s['mean']:>16.8f} "
            f"{s['median']:>18.8f} "
            f"{s['max']:>14.8f}"
        )
    results_dir = Path("results/phase4")
    results_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = results_dir / "phase4_ik_comparison.csv"

    with open(comparison_path, "w") as f:
        f.write("method,mean_ee_error,median_ee_error,max_ee_error\n")
        for s in summaries:
            f.write(
                f"{s['method']},"
                f"{s['mean']:.10f},"
                f"{s['median']:.10f},"
                f"{s['max']:.10f}\n"
            )

    print(f"\nSaved comparison table to: {comparison_path}")


if __name__ == "__main__":
    main()
