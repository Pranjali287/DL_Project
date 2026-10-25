"""
Publication-Quality Plotting Suite for GRAIL-NAV.
Generates research training curves, moving averages, and generalization comparisons.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Aesthetic research theme styling
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})


def plot_training_curves(df_fixed: pd.DataFrame, df_diverse: pd.DataFrame) -> Path:
    """Plot raw and moving-average training rewards for both agents."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Fixed-Map Training Curves
    axes[0].plot(df_fixed["episode"], df_fixed["reward"], color="#90caf9", alpha=0.35, label="Raw Episode Reward")
    axes[0].plot(df_fixed["episode"], df_fixed["rolling_reward_50"], color="#1565c0", linewidth=2.2, label="Moving Avg (50 eps)")
    axes[0].set_title("Fixed-Map Training: Reward Progression")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative Episode Reward")
    axes[0].legend(loc="lower right")

    # 2. Diverse-Map Training Curves
    axes[1].plot(df_diverse["episode"], df_diverse["reward"], color="#a5d6a7", alpha=0.35, label="Raw Episode Reward")
    axes[1].plot(df_diverse["episode"], df_diverse["rolling_reward_50"], color="#2e7d32", linewidth=2.2, label="Moving Avg (50 eps)")
    axes[1].set_title("Diverse-Map Training: Reward Progression")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Cumulative Episode Reward")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    out_path = PLOTS_DIR / "plot_training_curves.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_success_progression(df_fixed: pd.DataFrame, df_diverse: pd.DataFrame) -> Path:
    """Plot moving average success rate over training episodes."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df_fixed["episode"],
        df_fixed["rolling_success_rate_50"] * 100.0,
        color="#1976d2",
        linewidth=2.2,
        label="Fixed-Map Agent"
    )
    ax.plot(
        df_diverse["episode"],
        df_diverse["rolling_success_rate_50"] * 100.0,
        color="#388e3c",
        linewidth=2.2,
        linestyle="--",
        label="Diverse-Map Agent"
    )

    ax.set_title("Training Success Rate Progression (50-Episode Rolling Window)")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_path = PLOTS_DIR / "plot_success_progression.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_generalization_comparison(df_eval: pd.DataFrame) -> Path:
    """Plot bar chart comparing Success and Collision rates on Familiar vs Unseen maps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    agents = ["Fixed-Map Agent", "Diverse-Map Agent"]
    conditions = ["Familiar Map", "Unseen Maps"]
    x = np.arange(len(agents))
    width = 0.35

    # Extract success rates
    success_familiar = [
        df_eval[(df_eval["Agent Policy"] == ag) & (df_eval["Environment Condition"] == "Familiar Map")]["Success Rate (%)"].values[0]
        for ag in agents
    ]
    success_unseen = [
        df_eval[(df_eval["Agent Policy"] == ag) & (df_eval["Environment Condition"] == "Unseen Maps")]["Success Rate (%)"].values[0]
        for ag in agents
    ]

    # Success rate plot
    bars1 = ax1.bar(x - width/2, success_familiar, width, label="Familiar Map", color="#42a5f5", edgecolor="#1565c0")
    bars2 = ax1.bar(x + width/2, success_unseen, width, label="Unseen Maps", color="#ffa726", edgecolor="#e65100")
    ax1.set_title("Success Rate Comparison: Generalization Test")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents)
    ax1.set_ylim(0, 115)
    ax1.legend()

    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Collision rate plot
    collision_familiar = [
        df_eval[(df_eval["Agent Policy"] == ag) & (df_eval["Environment Condition"] == "Familiar Map")]["Collision Rate (%)"].values[0]
        for ag in agents
    ]
    collision_unseen = [
        df_eval[(df_eval["Agent Policy"] == ag) & (df_eval["Environment Condition"] == "Unseen Maps")]["Collision Rate (%)"].values[0]
        for ag in agents
    ]

    bars3 = ax2.bar(x - width/2, collision_familiar, width, label="Familiar Map", color="#ef5350", edgecolor="#b71c1c")
    bars4 = ax2.bar(x + width/2, collision_unseen, width, label="Unseen Maps", color="#ab47bc", edgecolor="#4a148c")
    ax2.set_title("Collision Rate Comparison (Safety Profile)")
    ax2.set_ylabel("Collision Rate (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(agents)
    ax2.set_ylim(0, 115)
    ax2.legend()

    for bar in bars3 + bars4:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = PLOTS_DIR / "plot_generalization_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_path_length_comparison(df_eval: pd.DataFrame) -> Path:
    """Plot average successful path length across conditions."""
    fig, ax = plt.subplots(figsize=(8, 5))

    agents = ["Fixed-Map Agent", "Diverse-Map Agent"]
    x = np.arange(len(agents))
    width = 0.35

    paths_familiar = [
        df_eval[(df_eval["Agent Policy"] == ag) & (df_eval["Environment Condition"] == "Familiar Map")]["Avg Succ Path Length"].values[0]
        for ag in agents
    ]
    paths_unseen = [
        df_eval[(df_eval["Agent Policy"] == ag) & (df_eval["Environment Condition"] == "Unseen Maps")]["Avg Succ Path Length"].values[0]
        for ag in agents
    ]

    b1 = ax.bar(x - width/2, paths_familiar, width, label="Familiar Map", color="#26a69a", edgecolor="#004d40")
    b2 = ax.bar(x + width/2, paths_unseen, width, label="Unseen Maps", color="#ab47bc", edgecolor="#4a148c")

    ax.set_title("Average Successful Path Length Comparison")
    ax.set_ylabel("Mean Steps to Goal")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.legend()

    for b in b1 + b2:
        val = b.get_height()
        if val > 0:
            ax.text(b.get_x() + b.get_width()/2.0, val + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = PLOTS_DIR / "plot_path_length_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def generate_all_plots() -> List[Path]:
    """Generate all research plots from recorded CSV files."""
    fixed_csv = RESULTS_DIR / "training_fixed.csv"
    diverse_csv = RESULTS_DIR / "training_diverse.csv"
    eval_csv = RESULTS_DIR / "evaluation_summary.csv"

    generated = []
    if fixed_csv.exists() and diverse_csv.exists():
        df_fixed = pd.read_csv(fixed_csv)
        df_diverse = pd.read_csv(diverse_csv)
        p1 = plot_training_curves(df_fixed, df_diverse)
        p2 = plot_success_progression(df_fixed, df_diverse)
        generated.extend([p1, p2])

    if eval_csv.exists():
        df_eval = pd.read_csv(eval_csv)
        p3 = plot_generalization_comparison(df_eval)
        p4 = plot_path_length_comparison(df_eval)
        generated.extend([p3, p4])

    print(f"[+] Successfully generated {len(generated)} research figures in: {PLOTS_DIR}")
    return generated
