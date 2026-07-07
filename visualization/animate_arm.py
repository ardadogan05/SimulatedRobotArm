import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from arms.planar_2link import forward_kinematics


def animate_arm(angle_sequence, L1=1.0, L2=1.0, interval=50):
    reach = L1 + L2

    fig, ax = plt.subplots()

    (line,) = ax.plot([], [], marker="o")

    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2-link planar arm animation")

    def update(frame):
        theta1, theta2 = angle_sequence[frame]

        base, joint1, end_effector = forward_kinematics(
            theta1,
            theta2,
            L1=L1,
            L2=L2,
        )

        xvalues = [base[0], joint1[0], end_effector[0]]
        yvalues = [base[1], joint1[1], end_effector[1]]

        line.set_data(xvalues, yvalues)

        return line,

    animation = FuncAnimation(
        fig,
        update,
        frames=len(angle_sequence),
        interval=interval,
        blit=True,
    )
    plt.show()

    return animation
