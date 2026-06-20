from solvers.analytical_2link import analytical_ik_2link
from solvers.jacobian_ik_2link import numerical_solver_2link
import numpy as np


def print_result(target, result, analytical):
    print("target:", target)
    print("analytical:", "reachable" if analytical is not None else "unreachable") #analytical returns None if no solution, uses this
    print("numerical success:", result["success"])
    print("numerical final error:", result["final_error"])
    print("numerical iterations:", result["iterations"])
    print("numerical theta:", np.round(result["theta"], 4))
    print()

def main():
    targets = [
        (1.0, 1.0),
        (1.5, 0.5),
        (-1.0, 1.0),
        (0.2, 1.5),
        (1.9, 0.1),
        (2.0, 0.0),
        (2.1, 0.0),
    ]

    print("=== Fixed initial guess ===")
    for target in targets:
        analytical = analytical_ik_2link(target)
        numerical = numerical_solver_2link(target)

        print_result(target, numerical, analytical)

    print("=== Different initial guesses ===")
    target = (1.0, 1.0)

    initial_guesses = [
        (0.1, 0.1),
        (1.0, 1.0),
        (-1.0, 1.0),
        (2.5, -2.5),
        (0.0, 0.0),
    ]

    for initial_theta in initial_guesses:
        result = numerical_solver_2link(target, initial_theta=initial_theta)

        print("initial theta:", initial_theta)
        print("success:", result["success"])
        print("final error:", result["final_error"])
        print("iterations:", result["iterations"])
        print("theta:", np.round(result["theta"], 4))
        print()


if __name__ == "__main__":
    main()