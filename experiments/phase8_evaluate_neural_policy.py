from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arms.planar_3link import forward_kinematics_3link
from models.neural_ik_3link import NeuralIK3Link
from solvers.jacobian_ik_3link import dls_solver_3link, numerical_solver_3link
from solvers.neural_ik_3link import neural_solver_3link
from training.torch_utils import get_torch_device
from visualization.animate_arm import animate_3link_solver_comparison


def print_result(result, target, tolerance):
    print("Single neural-policy rollout")
    print("----------------------------")
    print(f"Target:       {target}")
    print(f"Final theta:  {result['theta']}")
    print(f"Success:      {result['success']}")
    print(f"Tolerance:    {tolerance}")
    print(f"Final error:  {result['final_error']:.8f}")
    print(f"Policy steps: {result['iterations']}")
    print()


def plot_rollout(result, target, output_path):
    end_effector_history = []

    for theta in result["theta_history"]:
        end_effector = forward_kinematics_3link(
            theta[0],
            theta[1],
            theta[2],
        )[3]
        end_effector_history.append(end_effector)

    end_effector_history = np.array(end_effector_history)

    final_theta = result["theta"]
    final_arm = forward_kinematics_3link(
        final_theta[0],
        final_theta[1],
        final_theta[2],
    )
    final_arm = np.array(final_arm)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].plot(
        end_effector_history[:, 0],
        end_effector_history[:, 1],
        marker="o",
        markersize=3,
        label="End-effector path",
    )
    axes[0].plot(
        final_arm[:, 0],
        final_arm[:, 1],
        marker="o",
        linewidth=3,
        label="Final arm",
    )
    axes[0].scatter(target[0], target[1], marker="x", s=100, label="Target")
    axes[0].set_xlim(-3.2, 3.2)
    axes[0].set_ylim(-3.2, 3.2)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Neural Policy Rollout")
    axes[0].grid(True)
    axes[0].legend()

    steps = np.arange(len(result["error_history"]))
    axes[1].plot(steps, result["error_history"])
    axes[1].set_xlabel("Policy step")
    axes[1].set_ylabel("End-effector error")
    axes[1].set_title("Error During Rollout")
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def summarize_results(name, errors, successes, iterations, solve_times):
    errors = np.array(errors)

    return {
        "method": name,
        "success_rate": 100 * successes / len(errors),
        "mean_error": np.mean(errors),
        "median_error": np.median(errors),
        "max_error": np.max(errors),
        "mean_iterations": np.mean(iterations),
        "mean_time_ms": 1000 * np.mean(solve_times),
    }


def compare_methods(model, X_test, tolerance, max_iterations):
    #Lists to keep track of errors and other results
    pseudoinverse_errors = []
    dls_errors = []
    neural_errors = []

    pseudoinverse_successes = 0
    dls_successes = 0
    neural_successes = 0

    pseudoinverse_iterations = []
    dls_iterations = []
    neural_iterations = []

    pseudoinverse_times = []
    dls_times = []
    neural_times = []

    #Same as the 2-link comparison: use 1000 samples from the test dataset
    num_samples = min(1000, len(X_test))

    for i in range(num_samples):
        #The 3-link dataset also contains the current angles for the policy input
        target = X_test[i, 0:2]
        initial_theta = X_test[i, 2:5]

        # Pseudoinverse IK
        start_time = perf_counter()
        pseudoinverse_result = numerical_solver_3link(
            target,
            initial_theta=initial_theta,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        pseudoinverse_times.append(perf_counter() - start_time)

        pseudoinverse_successes += pseudoinverse_result["success"]
        pseudoinverse_errors.append(pseudoinverse_result["final_error"])
        pseudoinverse_iterations.append(pseudoinverse_result["iterations"])

        # Damped Least Squares IK
        start_time = perf_counter()
        dls_result = dls_solver_3link(
            target,
            initial_theta=initial_theta,
            damping=0.1,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        dls_times.append(perf_counter() - start_time)

        dls_successes += dls_result["success"]
        dls_errors.append(dls_result["final_error"])
        dls_iterations.append(dls_result["iterations"])

        # Neural IK policy rollout
        start_time = perf_counter()
        neural_result = neural_solver_3link(
            model,
            target,
            initial_theta=initial_theta,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        neural_times.append(perf_counter() - start_time)

        neural_successes += neural_result["success"]
        neural_errors.append(neural_result["final_error"])
        neural_iterations.append(neural_result["iterations"])

    summaries = [
        summarize_results(
            "Pseudoinverse IK",
            pseudoinverse_errors,
            pseudoinverse_successes,
            pseudoinverse_iterations,
            pseudoinverse_times,
        ),
        summarize_results(
            "DLS IK",
            dls_errors,
            dls_successes,
            dls_iterations,
            dls_times,
        ),
        summarize_results(
            "Neural policy",
            neural_errors,
            neural_successes,
            neural_iterations,
            neural_times,
        ),
    ]

    print("Final 3-link IK comparison")
    print("--------------------------")
    print(f"Samples: {num_samples} | tolerance: {tolerance} | max steps: {max_iterations}")
    print("Timing runs one complete IK solve at a time for every method.")
    print()
    print(
        f"{'Method':<18} "
        f"{'Success':>9} "
        f"{'Mean error':>12} "
        f"{'Median':>10} "
        f"{'Max error':>12} "
        f"{'Mean steps':>11} "
        f"{'Mean ms':>10}"
    )

    for summary in summaries:
        print(
            f"{summary['method']:<18} "
            f"{summary['success_rate']:>8.1f}% "
            f"{summary['mean_error']:>12.6f} "
            f"{summary['median_error']:>10.6f} "
            f"{summary['max_error']:>12.6f} "
            f"{summary['mean_iterations']:>11.2f} "
            f"{summary['mean_time_ms']:>10.3f}"
        )


def main():
    dataset_path = Path("data/processed/ik_3link_dataset.npz")
    model_path = Path("models/saved/neural_ik_3link.pt")
    output_path = Path("results/phase8/neural_policy_rollout.png")
    animation_path = Path("results/phase8/ik_3link_solver_comparison.gif")

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Generate the 3-link dataset before running the comparison"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            "Train the 3-link neural policy before running rollout evaluation"
        )

    device = get_torch_device()
    print("Using device:", device)
    print()

    model = NeuralIK3Link().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data = np.load(dataset_path)
    X_test = data["X_test"]

    tolerance = 0.05
    max_iterations = 100

    target = np.array([1.5, 1.5])
    initial_theta = np.array([0.1, 0.1, 0.1])

    result = neural_solver_3link(
        model,
        target,
        initial_theta=initial_theta,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    print_result(result, target, tolerance)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_rollout(result, target, output_path)
    print(f"Saved rollout plot to: {output_path}")
    print()

    pseudoinverse_result = numerical_solver_3link(
        target,
        initial_theta=initial_theta,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    dls_result = dls_solver_3link(
        target,
        initial_theta=initial_theta,
        damping=0.1,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    animation_results = {
        "Pseudoinverse IK": pseudoinverse_result,
        "DLS IK": dls_result,
        "Neural policy": result,
    }
    animate_3link_solver_comparison(
        animation_results,
        target,
        animation_path,
    )
    print(f"Saved solver animation to: {animation_path}")
    print()

    compare_methods(
        model,
        X_test,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


if __name__ == "__main__":
    main()
