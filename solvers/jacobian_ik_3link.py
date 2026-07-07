import numpy as np
from arms.planar_3link import forward_kinematics_3link
from solvers.utils import wrap_to_pi
#extremely similar to 2 link

def jacobian_3link(theta1, theta2, theta3, L1 = 1.0, L2 = 1.0, L3 = 1.0):
    t12 = theta1 + theta2
    t123 = theta1 + theta2 + theta3

    dxtheta1 = -L1*np.sin(theta1) - L2*np.sin(t12) - L3*np.sin(t123)
    dxtheta2 = -L2*np.sin(t12) - L3*np.sin(t123)
    dxtheta3 = -L3*np.sin(t123)

    dytheta1 = L1*np.cos(theta1) + L2*np.cos(t12) + L3*np.cos(t123)
    dytheta2 = L2*np.cos(t12) + L3*np.cos(t123)
    dytheta3 = L3*np.cos(t123)

    return np.array([[dxtheta1, dxtheta2, dxtheta3], [dytheta1, dytheta2, dytheta3]])

def numerical_solver_3link(target, L1 = 1.0, L2 = 1.0, L3 = 1.0, initial_theta = None):
    target = np.array(target, dtype=float)#incase input for target is an integer or not numpy array
    # If no initial guess is given, use a small default angle. If the user gives one, use that instead.
    if initial_theta is None:
        initial_theta = [0.1, 0.1, 0.1]
    theta = np.array(initial_theta, dtype = float) #inital guesses inside.
    tolerance = 1e-5
    learning_rate = 0.5 #To avoid overshoot
    max_iterations = 200 # more complex problem, therefore more iterations to compute

    for i in range(max_iterations):
        current_position = forward_kinematics_3link(theta[0], theta[1], theta[2], L1 = L1, L2 = L2, L3 = L3)[3] #only interested in end effector position
        error = target - current_position
        error_norm = np.linalg.norm(error)

        if error_norm < tolerance:
            return {
                "theta": theta,
                "success": True,
                "final_error": error_norm,
                "iterations": i + 1,
            } #dict to keep track of results
        
        J = jacobian_3link(theta[0], theta[1], theta[2], L1 = L1, L2 = L2, L3 = L3)
        delta_theta = np.linalg.pinv(J) @ error
        theta = theta + learning_rate*delta_theta
        theta = wrap_to_pi(theta)
    
    return {
        "theta": theta,
        "success": False,
        "final_error": error_norm,
        "iterations": max_iterations,
    } #dict to keep track of results

def dls_solver_3link(target, L1 = 1.0, L2 = 1.0, L3 = 1.0, initial_theta = None, damping = 0.1):
    target = np.array(target, dtype=float)

    if initial_theta is None:
        initial_theta = [0.1, 0.1, 0.1]

    theta = np.array(initial_theta, dtype=float)

    tolerance = 1e-5
    learning_rate = 0.5
    max_iterations = 200

    for i in range(max_iterations):
        current_position = forward_kinematics_3link(theta[0], theta[1], theta[2], L1 = L1, L2 = L2, L3 = L3)[3]
        error = target - current_position
        error_norm = np.linalg.norm(error)

        if error_norm < tolerance:
            return {
                "theta": theta,
                "success": True,
                "final_error": error_norm,
                "iterations": i + 1,
            }
        
        J = jacobian_3link(theta[0], theta[1], theta[2], L1 = L1, L2 = L2, L3 = L3)
        I = np.eye(2)
        delta_theta = J.T @ np.linalg.inv(J @ J.T + damping**2 * I) @ error

        theta = theta + learning_rate * delta_theta
        theta = wrap_to_pi(theta)

    _, _, _, current_position = forward_kinematics_3link(
        theta[0],
        theta[1],
        theta[2],
        L1=L1,
        L2=L2,
        L3=L3,
    )

    final_error = np.linalg.norm(target - current_position)

    return {
        "theta": theta,
        "success": False,
        "final_error": final_error,
        "iterations": max_iterations,
    }

