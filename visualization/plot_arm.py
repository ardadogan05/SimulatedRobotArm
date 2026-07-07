import matplotlib.pyplot as plt


def plot_arm(xvalues, yvalues, reach, target=None):
    if target is not None:
        plt.scatter(target[0], target[1], marker="x", label="Target")
        plt.legend()
    plt.plot(xvalues, yvalues, marker="o")
    plt.xlim(-reach, reach)
    plt.ylim(-reach, reach)
    plt.gca().set_aspect("equal", adjustable="box") #Keeping axis consistent.
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2-link planar arm")
    plt.show()

