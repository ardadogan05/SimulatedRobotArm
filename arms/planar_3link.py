import numpy as np

def forward_kinematics_3link(theta1, theta2, theta3, L1 = 1.0, L2 = 1.0, L3 = 1.0):
    base = np.array([0.0, 0.0])

    joint1 = np.array([
        L1 * np.cos(theta1),
        L1 * np.sin(theta1)
    ])

    joint2 = joint1 + np.array([
        L2 * np.cos(theta1 + theta2),
        L2 * np.sin(theta1 + theta2)
    ])

    end_effector = joint2 + np.array([
        L3 * np.cos(theta1 + theta2 + theta3),
        L3 * np.sin(theta1 + theta2 + theta2)
    ])