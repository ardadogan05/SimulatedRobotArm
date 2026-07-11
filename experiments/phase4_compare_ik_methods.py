from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from arms.planar_2link import forward_kinematics
from models.neural_ik_2link import NeuralIK2Link
from solvers.analytical_2link import analytical_ik_2link
from solvers.jacobian_ik_2link import numerical_solver_2link
from training.torch_utils import get_torch_device, synchronize_torch_device


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


def summarize_errors(name, errors, total_time, num_samples):
    errors = np.array(errors)

    return {
        "method": name,
        "mean": np.mean(errors),
        "median": np.median(errors),
        "max": np.max(errors),
        "mean_time_ms": 1000 * total_time / num_samples,
    }


def main():
    #path for the data used for testing
    dataset_path = Path("data/processed/ik_2link_dataset.npz")
    model_path = Path("models/saved/neural_ik_2link.pt") #model that is used

    data = np.load(dataset_path)

    X_test = data["X_test"]

    #Uses gpu if availabe, cpu otherwise
    device = get_torch_device()

    #loads model and puts in eval mode.
    model = NeuralIK2Link().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    #List to keep track of errors
    analytical_errors = []
    numerical_errors = []
    neural_errors = []

    analytical_total_time = 0.0
    numerical_total_time = 0.0

    # Start with 1000 samples because numerical IK is slower than analytical/neural.
    num_samples = 1000

    #The NN can process all 1000 targets together as one batch
    X_test_tensor = torch.tensor(
        X_test[:num_samples],
        dtype=torch.float32,
        device=device,
    )

    #One warm-up call avoids including PyTorch's first-call setup in the timing
    with torch.no_grad():
        model(X_test_tensor[:1])

    synchronize_torch_device(device)
    neural_start_time = perf_counter()
    with torch.no_grad():
        Y_pred = model(X_test_tensor).cpu().numpy()
    synchronize_torch_device(device)
    neural_total_time = perf_counter() - neural_start_time

    for i in range(num_samples):
        target_xy = X_test[i]
        neural_theta = Y_pred[i]

        # Analytical IK
        start_time = perf_counter()
        analytical_solutions = analytical_ik_2link(target_xy)
        theta_analytical = choose_positive_theta2_solution(analytical_solutions)
        analytical_total_time += perf_counter() - start_time

        if theta_analytical is not None:
            analytical_errors.append(
                end_effector_error(theta_analytical, target_xy)
            )

        # Numerical IK
        initial_theta = np.array([0.0, 0.0])

        start_time = perf_counter()
        numerical_result = numerical_solver_2link(
            target_xy,
            initial_theta=initial_theta,
        )
        numerical_total_time += perf_counter() - start_time

        theta_numerical = numerical_result["theta"]

        numerical_errors.append(
            end_effector_error(theta_numerical, target_xy)
        )

        # Neural IK
        neural_errors.append(
            end_effector_error(neural_theta, target_xy)
        )

    summaries = [
        summarize_errors(
            "Analytical IK",
            analytical_errors,
            analytical_total_time,
            num_samples,
        ),
        summarize_errors(
            "Numerical IK",
            numerical_errors,
            numerical_total_time,
            num_samples,
        ),
        summarize_errors(
            "Neural IK",
            neural_errors,
            neural_total_time,
            num_samples,
        ),
    ]

    #For printing a nice table
    print()
    print("Phase 4 IK method comparison")
    print("----------------------------")
    print(
        f"{'Method':<18} "
        f"{'Mean EE error':>16} "
        f"{'Median EE error':>18} "
        f"{'Max EE error':>14} "
        f"{'Mean ms':>10}"
    )

    for s in summaries:
        print(
            f"{s['method']:<18} "
            f"{s['mean']:>16.8f} "
            f"{s['median']:>18.8f} "
            f"{s['max']:>14.8f} "
            f"{s['mean_time_ms']:>10.6f}"
        )

    print("\nNeural IK timing uses one batch containing all 1000 targets.")
    results_dir = Path("results/phase4")
    results_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = results_dir / "phase4_ik_comparison.csv"

    with open(comparison_path, "w") as f:
        f.write("method,mean_ee_error,median_ee_error,max_ee_error,mean_time_ms\n")
        for s in summaries:
            f.write(
                f"{s['method']},"
                f"{s['mean']:.10f},"
                f"{s['median']:.10f},"
                f"{s['max']:.10f},"
                f"{s['mean_time_ms']:.10f}\n"
            )

    print(f"\nSaved comparison table to: {comparison_path}")


if __name__ == "__main__":
    main()
