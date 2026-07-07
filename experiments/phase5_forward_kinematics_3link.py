import numpy as np
from visualization.plot_arm import plot_arm
from arms.planar_3link import forward_kinematics_3link



def main():
    L1 = 1.0     
    L2 = 1.0
    L3 = 1.0
    
    base, joint1, joint2, end_effector = forward_kinematics_3link(
        0,
        np.pi / 4,
        np.pi / 2,
        L1 = L1,
        L2 = L2,
        L3 = L3,
        )
    
    target = np.array([1.0, 1.0]) # Example target marker, not necessarily reached.

    xvalues = [base[0], joint1[0], joint2[0], end_effector[0]]
    yvalues = [base[1], joint1[1], joint2[1], end_effector[1]]

    print(f"End effector position: {end_effector}")
    
    reach = L1 + L2 + L3
    plot_arm(xvalues, yvalues, reach, target = target)


if __name__ == "__main__":
    main()
