from solvers.analytical_2link import analytical_ik_2link
from arms.planar_2link import forward_kinematics
import numpy as np
#Function that check if the analyical solution of IK falls within the margin of error when compared with FK

def test_reachable_targets_reach_correct_position():
    targets = [
        (1, 1),
        (2, 0),
        (0, 0),
        (-1, 1),
    ]

    for target in targets:
        solutions = analytical_ik_2link(target)

        assert solutions is not None

        for theta1, theta2 in solutions:
            _, _, end_effector = forward_kinematics(theta1, theta2)

            error = np.linalg.norm(np.array(end_effector) - np.array(target))

            assert error < 1e-9 #stops and throws error unless criteria is met. 


def test_unreachable_target_returns_none():
    target = (2.1, 0)

    solutions = analytical_ik_2link(target)

    assert solutions is None