import numpy as np

def forward_kinematics(theta1, theta2, L1 = 1.0, L2 = 1.0):
    base = np.array([0.0, 0.0])

    joint1 = np.array([
        L1 * np.cos(theta1),
        L1 * np.sin(theta1)
    ])

    arm_end = joint1 + np.array([
        L2 * np.cos(theta1 + theta2),
        L2 * np.sin(theta1 + theta2)
    ])

    return base, joint1, arm_end