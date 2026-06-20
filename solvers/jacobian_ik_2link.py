import numpy as np
from arms.planar_2link import forward_kinematics



def jacobian_2link(theta1, theta2, L1 = 1.0, L2 = 1.0):
    dxtheta1 = -L1*np.sin(theta1) - L2*np.sin(theta1 + theta2)
    dxtheta2 = -L2*np.sin(theta1 + theta2)

    dytheta1 = L1*np.cos(theta1) + L2*np.cos(theta1 + theta2)
    dytheta2 = L2*np.cos(theta1 + theta2)

    return np.array([[dxtheta1, dxtheta2],[dytheta1, dytheta2]])

def wrap_to_pi(theta): #to avoid angles > 360 deg or 2 pi
    return (theta + np.pi) % (2 * np.pi) - np.pi 


def numerical_solver_2link(target, L1 = 1.0, L2 = 1.0):
    target = np.array(target, dtype = float) #incase input for target is an integer or not numpy array
    theta = np.array([0.1, 0.1]) #inital guesses inside.
    tolerance = 1e-5
    learning_rate = 0.5 #To avoid overshoot
    max_iterations = 100

    for i in range(max_iterations):
        current_position = forward_kinematics(theta[0], theta[1], L1 = L1, L2 = L2)[2] #only interested in end effector position
        error = target - current_position
        error_norm = np.linalg.norm(error)

        if error_norm < tolerance:
            return {
                "theta": theta,
                "success": True,
                "final_error": error_norm,
                "iterations": i + 1,
            } #dict to keep track of results
        
        J = jacobian_2link(theta[0], theta[1], L1 = L1, L2 = L2)
        delta_theta = np.linalg.pinv(J) @ error
        theta = theta + learning_rate*delta_theta
        theta = wrap_to_pi(theta)
    
    return {
        "theta": theta,
        "success": False,
        "final_error": error_norm,
        "iterations": max_iterations,
    } #dict to keep track of results

target = np.array([1.0, 1.0])
theta = numerical_solver_2link(target)["theta"]

_, _, end = forward_kinematics(theta[0], theta[1])
print("theta raw:", theta)
print("theta degrees:", np.round(np.degrees(theta), 2))
print("end:", end)
print("error:", np.linalg.norm(end - target))
        