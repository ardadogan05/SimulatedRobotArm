import numpy as np
from solvers.jacobian_ik_3link import jacobian_3link, numerical_solver_3link, dls_solver_3link
from arms.planar_3link import forward_kinematics_3link


def test_jacobian_3link_shape():
    J = jacobian_3link(0.0, 0.0, 0.0)

    assert J.shape == (2, 3)


def test_jacobian_3link_straight_configuration():
    J = jacobian_3link(0.0, 0.0, 0.0)

    expected = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 2.0, 1.0],
    ])

    assert np.allclose(J, expected)


def test_numerical_solver_3link_reaches_target():
    target = np.array([1.5, 1.5])

    result = numerical_solver_3link(target)

    assert result["success"]
    assert result["final_error"] < 1e-5


def test_numerical_solver_3link_solution_matches_forward_kinematics():
    target = np.array([1.5, 1.5])

    result = numerical_solver_3link(target)
    theta = result["theta"]

    _, _, _, end_effector = forward_kinematics_3link(
        theta[0],
        theta[1],
        theta[2],
    )

    assert np.allclose(end_effector, target, atol=1e-5)

def test_dls_solver_3link_reaches_target():
    target = np.array([1.5, 1.5])

    result = dls_solver_3link(target)

    assert result["success"]
    assert result["final_error"] < 1e-5