import random

import numpy as np

from arms.planar_3link import forward_kinematics_3link
from solvers.jacobian_ik_3link import jacobian_3link


def generate_3link_dataset(N):
    X = np.zeros((N, 5))  # inputs: target x, target y, current theta
    Y = np.zeros((N, 3))  # targets: delta theta1, delta theta2, delta theta3

    damping = 0.1
    learning_rate = 0.5

    for i in range(N):
        theta1 = random.uniform(-np.pi, np.pi)
        theta2 = random.uniform(-np.pi, np.pi)
        theta3 = random.uniform(-np.pi, np.pi)

        #generate a reachable target inside the arm's workspace
        radius = 3.0 * np.sqrt(random.random())
        target_angle = random.uniform(-np.pi, np.pi)
        target = np.array(
            [
                radius * np.cos(target_angle),
                radius * np.sin(target_angle),
            ]
        )

        _, _, _, current_position = forward_kinematics_3link(
            theta1,
            theta2,
            theta3,
        )
        error = target - current_position

        if np.linalg.norm(error) < 1e-5:
            delta_theta = np.zeros(3)
        else:
            J = jacobian_3link(theta1, theta2, theta3)
            I = np.eye(2)
            delta_theta = J.T @ np.linalg.inv(J @ J.T + damping**2 * I) @ error
            delta_theta = learning_rate * delta_theta

        X[i] = [target[0], target[1], theta1, theta2, theta3]
        Y[i] = delta_theta
    return X, Y
