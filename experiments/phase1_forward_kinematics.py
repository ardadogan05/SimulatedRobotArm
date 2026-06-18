import sys
from pathlib import Path
import numpy as np
from visualization.plot_arm import plot_arm
from arms.planar_2link import forward_kinematics


def main():
    base, joint1, end_effector = forward_kinematics(0, np.pi / 4)
    target = np.array([1.0, 1.0])

    xvalues = [base[0], joint1[0], end_effector[0]]
    yvalues = [base[1], joint1[1], end_effector[1]]

    print(f"End effector position: {end_effector}")
    
    reach = 2.0
    plot_arm(xvalues, yvalues, reach, target = target)


if __name__ == "__main__":
    main()
