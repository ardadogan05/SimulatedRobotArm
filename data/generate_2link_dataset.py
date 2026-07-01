import random
import numpy as np
from arms.planar_2link import forward_kinematics

def generate_2link_dataset(N):
    X = np.zeros((N, 2))  # inputs: x, y
    Y = np.zeros((N, 2))  # targets: theta1, theta2

    for i in range(N):
        theta1 = random.uniform(-np.pi / 2, np.pi / 2) #restricting angles at first to give consistent solution for NN to learn
        theta2 = random.uniform(0, np.pi)
        _, _, end_effector = forward_kinematics(theta1, theta2)

        X[i] = [end_effector[0], end_effector[1]]
        Y[i] = [theta1, theta2]
    return X, Y


        