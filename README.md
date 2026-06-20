# SimultatedRobotArm
Notes of numerical IK: 
Numerical IK reached all tested reachable targets with final error around 1e-5. 
Unlike analytical IK, it required 17–23 iterations and its returned joint configuration depended on the initial guess. 
For an unreachable target, analytical IK rejected it immediately, while numerical IK ran until max iterations and failed with nonzero final error.