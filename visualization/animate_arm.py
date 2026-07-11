from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from arms.planar_2link import forward_kinematics
from arms.planar_3link import forward_kinematics_3link


def animate_arm(angle_sequence, L1=1.0, L2=1.0, interval=50):
    reach = L1 + L2

    fig, ax = plt.subplots()

    (line,) = ax.plot([], [], marker="o")

    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2-link planar arm animation")

    def update(frame):
        theta1, theta2 = angle_sequence[frame]

        base, joint1, end_effector = forward_kinematics(
            theta1,
            theta2,
            L1=L1,
            L2=L2,
        )

        xvalues = [base[0], joint1[0], end_effector[0]]
        yvalues = [base[1], joint1[1], end_effector[1]]

        line.set_data(xvalues, yvalues)

        return line,

    animation = FuncAnimation(
        fig,
        update,
        frames=len(angle_sequence),
        interval=interval,
        blit=True,
    )
    plt.show()

    return animation


def animate_3link_solver_comparison(
    results,
    target,
    output_path,
    interval=250,
):
    """Animate one comparison (kept as a backwards-compatible convenience)."""
    return animate_3link_solver_examples(
        [{"target": target, "results": results}],
        output_path,
        interval=interval,
    )


def animate_3link_solver_examples(examples, output_path, interval=250):
    """Animate multiple solver examples sequentially in a single GIF."""
    if not examples:
        raise ValueError("At least one solver example is required")

    method_names = list(examples[0]["results"].keys())
    frame_counts = [
        max(
            len(example["results"][method_name]["theta_history"])
            for method_name in method_names
        )
        for example in examples
    ]
    frame_offsets = np.cumsum([0, *frame_counts])

    fig, axes = plt.subplots(1, len(method_names), figsize=(15, 5))
    arm_lines = []
    path_lines = []
    target_markers = []
    title_texts = []

    for ax, method_name in zip(axes, method_names):
        (arm_line,) = ax.plot([], [], marker="o", linewidth=3)
        (path_line,) = ax.plot([], [], marker=".", alpha=0.6)
        target_marker = ax.scatter([], [], marker="x", s=100, label="Target")

        arm_lines.append(arm_line)
        path_lines.append(path_line)
        target_markers.append(target_marker)
        title_texts.append(ax.set_title(method_name))

        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

    def update(frame):
        updated_lines = []
        example_index = int(np.searchsorted(frame_offsets[1:], frame, side="right"))
        example_index = min(example_index, len(examples) - 1)
        example = examples[example_index]
        results = example["results"]
        target = np.asarray(example["target"])
        example_frame = frame - frame_offsets[example_index]

        for i, method_name in enumerate(method_names):
            result = results[method_name]
            theta_history = results[method_name]["theta_history"]
            current_frame = min(example_frame, len(theta_history) - 1)
            theta = theta_history[current_frame]

            arm_points = np.array(
                forward_kinematics_3link(
                    theta[0],
                    theta[1],
                    theta[2],
                )
            )

            end_effector_path = []
            for previous_theta in theta_history[: current_frame + 1]:
                end_effector = forward_kinematics_3link(
                    previous_theta[0],
                    previous_theta[1],
                    previous_theta[2],
                )[3]
                end_effector_path.append(end_effector)

            end_effector_path = np.array(end_effector_path)

            arm_lines[i].set_data(arm_points[:, 0], arm_points[:, 1])
            path_lines[i].set_data(
                end_effector_path[:, 0],
                end_effector_path[:, 1],
            )
            target_markers[i].set_offsets(target.reshape(1, 2))
            title_texts[i].set_text(
                f"Example {example_index + 1} - {method_name}\n"
                f"Success: {result['success']} | Steps: {result['iterations']}"
            )

            updated_lines.extend(
                [arm_lines[i], path_lines[i], target_markers[i], title_texts[i]]
            )

        return updated_lines

    animation = FuncAnimation(
        fig,
        update,
        frames=int(frame_offsets[-1]),
        interval=interval,
        blit=True,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        output_path,
        writer=PillowWriter(fps=1000 / interval),
    )
    plt.close(fig)

    return animation
