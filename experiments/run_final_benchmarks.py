"""Run both final IK benchmarks using the published trained weights."""

from experiments.phase4_compare_ik_methods import main as benchmark_2link
from experiments.phase8_evaluate_neural_policy import main as benchmark_3link


def main():
    print("Running final 2-link benchmark")
    print("================================")
    benchmark_2link(use_dataset=False)

    print("\nRunning final 3-link benchmark")
    print("================================")
    benchmark_3link(use_dataset=False)


if __name__ == "__main__":
    main()
