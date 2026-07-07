import numpy as np

from arms.planar_2link import forward_kinematics
from solvers.jacobian_ik_2link import jacobian_2link, numerical_solver_2link

#run "python -m pytest" in terminal to test.


def test_jacobian_at_stretched_configuration(): #test for jacobian matrix
    J = jacobian_2link(0.0, 0.0, L1=1.0, L2=1.0)

    expected = np.array(
        [
            [0.0, 0.0],
            [2.0, 1.0],
        ]
    )

    assert np.allclose(J, expected)


def test_numerical_ik_reaches_simple_target():
    target = np.array([1.0, 1.0])
    result = numerical_solver_2link(target)

    assert result["success"]

    theta = result["theta"]
    _, _, end_effector = forward_kinematics(theta[0], theta[1])
    error = np.linalg.norm(end_effector - target)

    assert error < 1e-5


def test_numerical_ik_reaches_multiple_targets():
    targets = [
        np.array([1.0, 1.0]),
        np.array([1.5, 0.5]),
        np.array([-1.0, 1.0]),
        np.array([0.2, 1.5]),
    ]

    for target in targets:
        result = numerical_solver_2link(target)

        assert result["success"]
        assert result["iterations"] > 0

        theta = result["theta"]
        _, _, end_effector = forward_kinematics(theta[0], theta[1])

        error = np.linalg.norm(end_effector - target)

        assert error < 1e-5
