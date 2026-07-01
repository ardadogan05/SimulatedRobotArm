import numpy as np
from visualization.animate_arm import animate_arm

# Experiment file for showing the animation.



def main():
    t = np.linspace(0,np.pi /3, 200)

    theta1 = np.sin(t)
    theta2 = np.sin(t)

    angle_sequence = list(zip(theta1, theta2))
    animate_arm(angle_sequence)

if __name__ == "__main__":
    main()
