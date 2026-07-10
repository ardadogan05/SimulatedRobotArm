import random

import numpy as np

from solvers.jacobian_ik_3link import dls_step_3link

#Idea for this NN is to give us delta theta, so that we get closer to target
#This is done repeatedly until we have reached the target, essentially mimicking each step of the DLS numerical method.


def generate_3link_dataset(N):
    X = np.zeros((N, 5))  # inputs: target x, target y, current theta
    Y = np.zeros((N, 3))  # targets: delta theta1, delta theta2, delta theta3

    for i in range(N):
        #current angles are included so the same target does not have many answers
        theta1 = random.uniform(-np.pi, np.pi)
        theta2 = random.uniform(-np.pi, np.pi)
        theta3 = random.uniform(-np.pi, np.pi)

        #sqrt spreads the targets evenly over the reachable area
        radius = 3.0 * np.sqrt(random.random())
        target_angle = random.uniform(-np.pi, np.pi)

        #target converted from polar to cartesian
        target = np.array(
            [
                radius * np.cos(target_angle),
                radius * np.sin(target_angle),
            ]
        )

        #only saves the next movement, not the final angles from the full solver
        delta_theta, _ = dls_step_3link(
            target,
            [theta1, theta2, theta3],
        )

        #NN learns what movement to make from the current angles towards the target
        X[i] = [target[0], target[1], theta1, theta2, theta3]
        Y[i] = delta_theta
    return X, Y
