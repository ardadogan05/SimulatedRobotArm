import random

import numpy as np

from solvers.jacobian_ik_3link import dls_step_3link
from solvers.utils import wrap_to_pi

#Idea for this NN is to give us delta theta, so that we get closer to target
#This is done repeatedly until we have reached the target, essentially mimicking each step of the DLS numerical method.


def generate_3link_dataset(N, max_steps_per_rollout=100, max_step_norm=0.5):
    X = np.zeros((N, 5))  # inputs: target x, target y, current theta
    Y = np.zeros((N, 3))  # targets: delta theta1, delta theta2, delta theta3

    sample = 0

    while sample < N:
        #Each rollout begins from random current angles and a random target
        theta = np.array(
            [
                random.uniform(-np.pi, np.pi),
                random.uniform(-np.pi, np.pi),
                random.uniform(-np.pi, np.pi),
            ]
        )

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

        #Save every step of the DLS trajectory, including small near-target steps
        for _ in range(max_steps_per_rollout):
            delta_theta, error_norm = dls_step_3link(target, theta)

            #Large updates are limited so both teacher and NN rollouts stay stable
            step_norm = np.linalg.norm(delta_theta)
            if step_norm > max_step_norm:
                delta_theta = delta_theta * max_step_norm / step_norm

            #NN learns what movement to make from the current angles towards the target
            X[sample] = [target[0], target[1], theta[0], theta[1], theta[2]]
            Y[sample] = delta_theta
            sample += 1

            if sample == N or error_norm < 1e-5:
                break

            theta = theta + delta_theta
            theta = wrap_to_pi(theta)

    return X, Y
