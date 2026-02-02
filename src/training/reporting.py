from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from src.training.artifacts import save_json


def save_date_plan(out_dir: Path, plan: Dict):
    save_json(out_dir / "date_plan.json", plan)


def write_report(path: Path, lines: Iterable[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    path.write_text(text, encoding="utf-8")


def _apply_plot_style(ax):
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _save_fig(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    fig.clear()


def plot_cv_best_iter(cv_df: pd.DataFrame, out_path: Path, title: str):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_subplot(111)
    ax.hist(cv_df["best_iter"].dropna().values, bins=12, color="#1F7A8C", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("best_iter")
    ax.set_ylabel("count")
    _apply_plot_style(ax)
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_holdout_binary(hold_df: pd.DataFrame, out_path: Path):
    import matplotlib.pyplot as plt

    metrics = ["hit_at_1", "hit_at_3", "hit_at_5"]
    fig = plt.figure(figsize=(7.6, 4.4))
    ax = fig.add_subplot(111)
    x = range(len(hold_df))
    width = 0.22

    colors = ["#1F7A8C", "#4FA3A5", "#97C1A9"]
    for i, m in enumerate(metrics):
        ax.bar([v + (i - 1) * width for v in x], hold_df[m].values, width=width, label=m, color=colors[i])

    ax.set_xticks(list(x))
    ax.set_xticklabels(hold_df["target"].values)
    ax.set_ylabel("score")
    ax.set_title("Holdout (2025) hit@k by target")
    ax.legend()
    _apply_plot_style(ax)
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_holdout_rank(hold: Dict, out_path: Path):
    import matplotlib.pyplot as plt

    keys = ["mrr_winner", "ndcg@3", "ndcg@5", "hit_at_1", "hit_at_3", "hit_at_5"]
    vals = [hold.get(k, 0.0) for k in keys]
    fig = plt.figure(figsize=(7.6, 4.2))
    ax = fig.add_subplot(111)
    ax.bar(keys, vals, color="#1F7A8C")
    ax.set_ylabel("score")
    ax.set_title("Holdout (2025) ranking metrics")
    ax.tick_params(axis="x", rotation=25)
    _apply_plot_style(ax)
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_cv_metrics_rank(cv_df: pd.DataFrame, out_path: Path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.6, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(cv_df["valid_end"], cv_df["ndcg@3"], marker="o", label="ndcg@3")
    ax.plot(cv_df["valid_end"], cv_df["ndcg@5"], marker="o", label="ndcg@5")
    ax.set_title("CV metrics by fold (rank)")
    ax.set_xlabel("valid_end")
    ax.set_ylabel("score")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    _apply_plot_style(ax)
    _save_fig(fig, out_path)
    plt.close(fig)
