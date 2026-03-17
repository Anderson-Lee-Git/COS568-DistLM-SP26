import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List

TASKS = ["task_2a", "task_2b", "task_3"]
TASK_LABELS = ["2A", "2B", "3"]
NUM_RANKS = 4
DATA_DIR = "./RTE"

# ICML-style color palette (colorblind-friendly)
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

# ICML-style rcParams
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "figure.dpi": 600,
})

NUM_COLS = len(TASKS) + 1  # +1 for the Combined column


def load_loss_curve(task: str, rank: int) -> List[float]:
    path = f"{DATA_DIR}/{task}/loss_curve_{rank}.json"
    with open(path) as f:
        return json.load(f)


def plot_loss_curves():
    fig, axes = plt.subplots(
        nrows=NUM_RANKS,
        ncols=NUM_COLS,
        figsize=(3.8 * NUM_COLS, 2.4 * NUM_RANKS),
        sharex=False,
        sharey=False,
    )

    for row in range(NUM_RANKS):
        # Per-task columns
        for col, (task, label) in enumerate(zip(TASKS, TASK_LABELS)):
            ax = axes[row][col]
            loss = load_loss_curve(task, row)
            ax.plot(range(len(loss)), loss, color=COLORS[col])
            ax.set_title(f"Task {label} — Rank {row}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")

        # Combined column (last)
        ax_combined = axes[row][NUM_COLS - 1]
        for col, (task, label) in enumerate(zip(TASKS, TASK_LABELS)):
            loss = load_loss_curve(task, row)
            ax_combined.plot(
                range(len(loss)),
                loss,
                color=COLORS[col],
                label=f"Task {label}",
            )
        ax_combined.set_title(f"Combined — Rank {row}")
        ax_combined.set_xlabel("Step")
        ax_combined.set_ylabel("Loss")
        ax_combined.legend(loc="upper right")

    # fig.suptitle("Training Loss Curves by Task and Rank", fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()
    plt.savefig("loss_curves.png", dpi=600, bbox_inches="tight")
    plt.show()


def load_training_time(task: str, rank: int) -> List[float]:
    path = f"{DATA_DIR}/{task}/training_time_{rank}.json"
    with open(path) as f:
        return json.load(f)


def latex_training_time_table() -> str:
    # Build a dict: avg_time[task][rank] = mean iteration time (ms), warmup step excluded
    avg_time: dict[str, dict[int, float]] = {}
    for task in TASKS:
        avg_time[task] = {}
        for rank in range(NUM_RANKS):
            times = load_training_time(task, rank)
            # Discard the first iteration (warmup / cache-cold outlier)
            times = times[1:]
            avg_time[task][rank] = sum(times) / len(times)

    # LaTeX table: rows = ranks, columns = tasks
    col_fmt = "l" + "r" * len(TASKS)
    task_headers = " & ".join(f"Task {lbl}" for lbl in TASK_LABELS)
    header = f"Rank & {task_headers} \\\\"

    rows = []
    for rank in range(NUM_RANKS):
        cells = " & ".join(f"{avg_time[task][rank]:.4f}" for task in TASKS)
        rows.append(f"{rank} & {cells} \\\\")

    table = "\n".join([
        "\\begin{table}[h]",
        "  \\centering",
        f"  \\begin{{tabular}}{{{col_fmt}}}",
        "    \\toprule",
        f"    {header}",
        "    \\midrule",
        *[f"    {row}" for row in rows],
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption{Average per-iteration training time (s) per rank and task."
        " The first iteration is excluded as a warmup step.}",
        "  \\label{tab:training_time}",
        "\\end{table}",
    ])
    print(table)
    return table

def main():
    plot_loss_curves()
    latex_training_time_table()


if __name__ == "__main__":
    main()
