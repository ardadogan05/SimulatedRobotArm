import numpy as np
from visualization.plot_arm import plot_arm
from arms.planar_2link import forward_kinematics


def main():
    L2 = 1.0
    L1 = 1.0 
    base, joint1, end_effector = forward_kinematics(0, np.pi / 4, L1 = L1, L2 = L2)
    target = np.array([1.0, 1.0]) # Example target marker, not necessarily reached.

    xvalues = [base[0], joint1[0], end_effector[0]]
    yvalues = [base[1], joint1[1], end_effector[1]]

    print(f"End effector position: {end_effector}")
    
    reach = L1 + L2
    plot_arm(xvalues, yvalues, reach, target = target)


if __name__ == "__main__":
    main()
