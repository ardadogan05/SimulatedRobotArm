import numpy as np
from arms.planar_3link import forward_kinematics_3link
from solvers.jacobian_ik_3link import numerical_solver_3link
from visualization.plot_arm import plot_arm


def main():
    target = np.array([1.333, 0.5])

    result = numerical_solver_3link(target)

    theta = result["theta"]

    base, joint1, joint2, end_effector = forward_kinematics_3link(
        theta[0],
        theta[1],
        theta[2],
    )

    xvalues = [base[0], joint1[0], joint2[0], end_effector[0]]
    yvalues = [base[1], joint1[1], joint2[1], end_effector[1]]

    reach = 3.0

    print("Phase 6: 3-link pseudoinverse IK")
    print("--------------------------------")
    print(f"Target:       {target}")
    print(f"Theta:        {theta}")
    print(f"End effector: {end_effector}")
    print(f"Success:      {result['success']}")
    print(f"Final error:  {result['final_error']:.8f}")
    print(f"Iterations:   {result['iterations']}")

    plot_arm(xvalues, yvalues, reach, target=target)


if __name__ == "__main__":
    main()