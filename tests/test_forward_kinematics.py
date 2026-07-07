import numpy as np

from arms.planar_2link import forward_kinematics

#run "python -m pytest" in terminal to test.


def test_forward_kinematics_stretched_arm():
    _, joint1, end_effector = forward_kinematics(0.0, 0.0, L1=1.0, L2=1.0)

    assert np.allclose(joint1, np.array([1.0, 0.0]))
    assert np.allclose(end_effector, np.array([2.0, 0.0]))


def test_forward_kinematics_right_angle_second_link():
    _, joint1, end_effector = forward_kinematics(0.0, np.pi / 2, L1=1.0, L2=1.0)

    assert np.allclose(joint1, np.array([1.0, 0.0]))
    assert np.allclose(end_effector, np.array([1.0, 1.0]))


def test_forward_kinematics_both_links_vertical():
    _, joint1, end_effector = forward_kinematics(np.pi / 2, 0.0, L1=1.0, L2=1.0)

    assert np.allclose(joint1, np.array([0.0, 1.0]))
    assert np.allclose(end_effector, np.array([0.0, 2.0]))
