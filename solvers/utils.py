import numpy as np

def wrap_to_pi(theta): #to avoid angles > 360 deg or 2 pi
    return (theta + np.pi) % (2 * np.pi) - np.pi 