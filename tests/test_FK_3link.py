import numpy as np
from arms.planar_3link import forward_kinematics_3link


def test_3link_straight_right():
    base, joint1, joint2, end_effector = forward_kinematics_3link(0, 0, 0)

    assert np.allclose(base, [0.0, 0.0])
    assert np.allclose(joint1, [1.0, 0.0])
    assert np.allclose(joint2, [2.0, 0.0])
    assert np.allclose(end_effector, [3.0, 0.0])


def test_3link_straight_up():
    base, joint1, joint2, end_effector = forward_kinematics_3link(
        np.pi / 2,
        0,
        0,
    )

    assert np.allclose(base, [0.0, 0.0], atol=1e-10)
    assert np.allclose(joint1, [0.0, 1.0], atol=1e-10)
    assert np.allclose(joint2, [0.0, 2.0], atol=1e-10)
    assert np.allclose(end_effector, [0.0, 3.0], atol=1e-10)


def test_3link_folded():
    base, joint1, joint2, end_effector = forward_kinematics_3link(
        0,
        np.pi,
        0,
    )

    assert np.allclose(base, [0.0, 0.0], atol=1e-10)
    assert np.allclose(joint1, [1.0, 0.0], atol=1e-10)
    assert np.allclose(joint2, [0.0, 0.0], atol=1e-10)
    assert np.allclose(end_effector, [-1.0, 0.0], atol=1e-10)
