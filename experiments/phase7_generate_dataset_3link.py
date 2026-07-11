from pathlib import Path

import numpy as np

from arms.planar_3link import forward_kinematics_3link
from data.dataset_utils import split_dataset
from data.generate_3link_dataset import generate_3link_dataset

#Generates and saves the training data, then checks that the DLS steps work.
#This indirectly checks that the generated input and target pairs match correctly.


def end_effector(theta):
    _, _, _, ee = forward_kinematics_3link(
        theta[0],
        theta[1],
        theta[2],
    )
    return ee


def main():
    # Generate dataset
    X, Y = generate_3link_dataset(50_000)

    # Split dataset
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_val shape:", X_val.shape)
    print("Y_val shape:", Y_val.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

    # Save split dataset
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    dataset_path = Path("data/processed/ik_3link_dataset.npz")
    np.savez(
        dataset_path,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=X_test,
        Y_test=Y_test,
    )

    improved = 0
    errors_before = []
    errors_after = []

    #Checking 1000 samples is enough for a quick sanity check
    num_sanity_samples = 1000

    for i in range(num_sanity_samples):
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

        #if delta_theta from dls solver leads us closer to target, improved, if not, not improved
        if error_after < error_before:
            improved += 1

    print("3-link policy dataset sanity check")
    print("----------------------------------")
    print(f"Samples tested: {num_sanity_samples}")
    print(f"Improved samples: {improved}")
    print(f"Improvement rate: {100 * improved / num_sanity_samples:.2f}%")
    print(f"Mean error before: {np.mean(errors_before):.6f}")
    print(f"Mean error after:  {np.mean(errors_after):.6f}")
    print(f"Saved dataset to: {dataset_path}")


if __name__ == "__main__":
    main()
