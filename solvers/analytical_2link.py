import numpy as np


def is_reachable(target, L1=1.0, L2=1.0):
    r = np.sqrt(target[0]**2 + target[1]**2)
    return (abs(L1 - L2) <= r) and (r <= (L1 + L2))


def theta2_solutions(target, L1=1.0, L2=1.0):
    x, y = target
    c = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    c = np.clip(c, -1.0, 1.0) #Avoid error due to usage of float.
    theta2_a = np.arccos(c)
    theta2_b = -np.arccos(c)
    return theta2_a, theta2_b


def theta1_solutions(target, L1=1.0, L2=1.0):
    x, y = target
    r = np.sqrt(x**2 + y**2)

    alpha = np.atan2(y, x)

    c = (L1**2 + r**2 - L2**2) / (2 * L1 * r)
    c = np.clip(c, -1.0, 1.0) #same as l 11
    beta = np.arccos(c)

    theta1_a = alpha - beta
    theta1_b = alpha + beta
    return theta1_a, theta1_b


def analytical_ik_2link(target, L1=1.0, L2=1.0):
    x, y = target
    r = np.sqrt(x**2 + y**2)

    if not is_reachable(target, L1, L2): #no point running if it's out of range
        return None

    if np.isclose(r, 0.0) and np.isclose(L1, L2):
        return [(0.0, np.pi)] #Standard solution if L1 = L2 to avoid divison by zero.

    theta1_a, theta1_b = theta1_solutions(target, L1, L2)
    theta2_a, theta2_b = theta2_solutions(target, L1, L2)
    return [(theta1_a, theta2_a), (theta1_b, theta2_b)]
