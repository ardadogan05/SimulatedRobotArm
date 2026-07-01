# SimulatedRobotArm
Notes of numerical IK:
Numerical IK reached all tested reachable targets with final error around 1e-5.
Unlike analytical IK, it required 17-23 iterations and its returned joint configuration depended on the initial guess.
For an unreachable target, analytical IK rejected it immediately, while numerical IK ran until max iterations and failed with nonzero final error.

Training neural network: Training and validation loss decreased and then plateaued around 10^-3, suggesting the small MLP learned the restricted 2-link inverse mapping reasonably well. Since train and validation loss remained close, there was no strong sign of overfitting. Further evaluation should focus on end-effector error rather than only angle MSE.
