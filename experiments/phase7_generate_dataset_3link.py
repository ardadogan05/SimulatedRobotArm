import numpy as np

from arms.planar_3link import forward_kinematics_3link
from data.generate_3link_dataset import generate_3link_dataset

#mainly tests that dls step works, but indirectly checks that the datageneration is valid.

def end_effector(theta):
    _, _, _, ee = forward_kinematics_3link(
        theta[0],
        theta[1],
        theta[2],
    )
    return ee


def main():
    X, Y = generate_3link_dataset(1000)

    improved = 0
    errors_before = []
    errors_after = []

    for i in range(len(X)):
        #cartesian coordinates of target
        target = X[i, 0:2]
        theta_before = X[i, 2:5]
        #delta_theta from DLS solver
        delta_theta = Y[i]

        theta_after = theta_before + delta_theta

        ee_before = end_effector(theta_before)
        ee_after = end_effector(theta_after)

        error_before = np.linalg.norm(target - ee_before)
        error_after = np.linalg.norm(target - ee_after)

        errors_before.append(error_before)
        errors_after.append(error_after)

        #if delta_theta from dls solver leads us closer to target, imporved, if not, not improved
        if error_after < error_before:
            improved += 1

    print("3-link policy dataset sanity check")
    print("----------------------------------")
    print(f"Samples tested: {len(X)}")
    print(f"Improved samples: {improved}")
    print(f"Improvement rate: {100 * improved / len(X):.2f}%")
    print(f"Mean error before: {np.mean(errors_before):.6f}")
    print(f"Mean error after:  {np.mean(errors_after):.6f}")


if __name__ == "__main__":
    main()