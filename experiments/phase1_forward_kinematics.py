import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow this example to run either as a module or as a direct script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arms.planar_2link import forward_kinematics


def main():
    base, joint1, arm_end = forward_kinematics(0, np.pi/2)

    xvalues = [base[0], joint1[0], arm_end[0]]
    yvalues = [base[1], joint1[1], arm_end[1]]

    print(f"End effector position: {arm_end}")
    
    reach = 2.0
    plt.plot(xvalues, yvalues, marker="o")
    plt.xlim(-reach, reach)
    plt.ylim(-reach, reach)
    plt.gca().set_aspect("equal", adjustable="box") #Keeping axis consistent.
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2-link planar arm")
    plt.show()


if __name__ == "__main__":
    main()
