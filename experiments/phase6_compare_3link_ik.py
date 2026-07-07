import numpy as np

from arms.planar_3link import forward_kinematics_3link
from solvers.jacobian_ik_3link import dls_solver_3link, numerical_solver_3link
from visualization.plot_arm import plot_arm


def print_result(method_name, result, target):
    theta = result["theta"]

    end_effector = forward_kinematics_3link(
        theta[0],
        theta[1],
        theta[2],
    )[3]

    print(method_name)
    print("-" * len(method_name))
    print(f"Target:       {target}")
    print(f"Theta:        {theta}")
    print(f"End effector: {end_effector}")
    print(f"Success:      {result['success']}")
    print(f"Final error:  {result['final_error']:.8f}")
    print(f"Iterations:   {result['iterations']}")
    print()


def plot_solution(result, target, title):
    theta = result["theta"]

    base, joint1, joint2, end_effector = forward_kinematics_3link(
        theta[0],
        theta[1],
        theta[2],
    )

    xvalues = [base[0], joint1[0], joint2[0], end_effector[0]]
    yvalues = [base[1], joint1[1], joint2[1], end_effector[1]]

    reach = 3.0

    print(f"Showing plot: {title}")
    plot_arm(xvalues, yvalues, reach, target=target)


def run_random_sanity_check(num_targets=100):
    np.random.seed(0)

    L1 = 1.0
    L2 = 1.0
    L3 = 1.0
    max_reach = L1 + L2 + L3

    pseudoinverse_successes = 0
    dls_successes = 0

    pseudoinverse_errors = []
    dls_errors = []

    pseudoinverse_iterations = []
    dls_iterations = []

    for _ in range(num_targets):
        #generates random radius within reach
        radius = max_reach * np.sqrt(np.random.rand())
        angle = np.random.uniform(-np.pi, np.pi)

        #target converted from polar to cartesian
        target = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        initial_theta = np.random.uniform(-np.pi, np.pi, size=3)

        pseudoinverse_result = numerical_solver_3link(
            target,
            initial_theta=initial_theta,
        )

        dls_result = dls_solver_3link(
            target,
            initial_theta=initial_theta,
            damping=0.1,
        )

        # adds success, error and total iterations to lists for later use
        pseudoinverse_successes += pseudoinverse_result["success"]
        dls_successes += dls_result["success"]

        pseudoinverse_errors.append(pseudoinverse_result["final_error"])
        dls_errors.append(dls_result["final_error"])

        pseudoinverse_iterations.append(pseudoinverse_result["iterations"])
        dls_iterations.append(dls_result["iterations"])

    print("Random target sanity check")
    print("--------------------------")
    print(f"Targets tested: {num_targets}")
    print()
    pseudoinverse_success_rate = 100 * pseudoinverse_successes / num_targets
    dls_success_rate = 100 * dls_successes / num_targets

    print(f"Pseudoinverse success rate: {pseudoinverse_success_rate:.1f}%")
    print(f"Pseudoinverse mean error:   {np.mean(pseudoinverse_errors):.8f}")
    print(f"Pseudoinverse mean iters:   {np.mean(pseudoinverse_iterations):.2f}")
    print()
    print(f"DLS success rate:           {dls_success_rate:.1f}%")
    print(f"DLS mean error:             {np.mean(dls_errors):.8f}")
    print(f"DLS mean iters:             {np.mean(dls_iterations):.2f}")
    print()


def main():
    target = np.array([1.5, 1.5])
    initial_theta = np.array([0.1, 0.1, 0.1])

    pseudoinverse_result = numerical_solver_3link(
        target,
        initial_theta=initial_theta,
    )

    dls_result = dls_solver_3link(
        target,
        initial_theta=initial_theta,
        damping=0.1,
    )

    print()
    print("Phase 6: 3-link IK comparison")
    print("=============================")
    print()

    print_result("Pseudoinverse IK", pseudoinverse_result, target)
    print_result("Damped Least Squares IK", dls_result, target)

    plot_solution(pseudoinverse_result, target, "Pseudoinverse IK")
    plot_solution(dls_result, target, "Damped Least Squares IK")

    run_random_sanity_check(num_targets=100)


if __name__ == "__main__":
    main()
