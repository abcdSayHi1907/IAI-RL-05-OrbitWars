from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
HISTORY_PATH = ROOT / "history.txt"
OUTPUT_DIR = ROOT / "history_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LINE_RE = re.compile(
    r"update=(?P<update>\d+)\s+"
    r"outcome=(?P<outcome>[+-]?\d+\.\d+)\s+"
    r"loss=(?P<loss>[+-]?\d+\.\d+)\s+"
    r"ev=(?P<ev>[+-]?\d+\.\d+)\s+"
    r"kl=(?P<kl>[+-]?\d+\.\d+)\s+"
    r"clipfrac=(?P<clipfrac>[+-]?\d+\.\d+)\s+"
    r"entropy=(?P<entropy>[+-]?\d+\.\d+)\s+"
    r"pass=(?P<pass_frac>[+-]?\d+\.\d+)\s+"
    r"optional_pass=(?P<optional_pass>[+-]?\d+\.\d+)\s+"
    r"baseline=(?P<baseline>[+-]?\d+\.\d+)\s+"
    r"top=(?P<top_name>[^:\s]+):(?P<top_frac>[+-]?\d+\.\d+)"
    r"(?:\s+eval_w/l/d=(?P<eval_win>\d+\.\d+)/"
    r"(?P<eval_loss>\d+\.\d+)/(?P<eval_draw>\d+\.\d+))?"
)


def parse_history(path: Path) -> pd.DataFrame:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        match = LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"Cannot parse line {line_no}: {line}")
        row = match.groupdict()
        for key in row:
            if key == "top_name" or row[key] is None:
                continue
            row[key] = float(row[key])
        row["update"] = int(row["update"])
        rows.append(row)
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return pd.DataFrame(rows).sort_values("update").reset_index(drop=True)


def rolling(series: pd.Series, window: int = 10) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Update")


def save_dashboard(df: pd.DataFrame) -> Path:
    updates = df["update"]
    eval_df = df.dropna(subset=["eval_win"])
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Whole-Plan PPO Training Metrics", fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(updates, df["ev"], color="#059669", linewidth=1.8, label="Explained variance")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Critic Fit")
    ax.set_ylim(min(-0.1, df["ev"].min() - 0.05), 1.05)
    ax.legend()
    style_axis(ax)

    ax = axes[0, 1]
    ax.plot(updates, df["entropy"], color="#db2777", alpha=0.6, label="Entropy")
    ax.plot(updates, rolling(df["entropy"]), color="#831843", linewidth=2.2, label="Rolling mean (10)")
    ax.set_title("Policy Exploration")
    ax.legend()
    style_axis(ax)

    ax = axes[1, 0]
    if not eval_df.empty:
        ax.plot(eval_df["update"], eval_df["eval_win"], marker="o", linewidth=2, label="Win")
        ax.plot(eval_df["update"], eval_df["eval_loss"], marker="o", linewidth=2, label="Loss")
        ax.plot(eval_df["update"], eval_df["eval_draw"], marker="o", linewidth=2, label="Draw")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="50%")
        for _, row in eval_df.iterrows():
            ax.annotate(
                f"{row['eval_win']:.2f}",
                (row["update"], row["eval_win"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax.set_title("Evaluation vs Baseline")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    style_axis(ax)

    ax = axes[1, 1]
    ax.plot(updates, df["kl"], color="#7c3aed", label="Approx. KL")
    ax.axhline(0.03, color="#dc2626", linestyle="--", linewidth=1.2, label="KL warning 0.03")
    ax2 = ax.twinx()
    ax2.plot(updates, df["clipfrac"], color="#f59e0b", alpha=0.85, label="Clip fraction")
    ax2.axhline(0.30, color="#f59e0b", linestyle=":", linewidth=1.2, label="Clip warning 0.30")
    ax.set_title("PPO Update Magnitude")
    ax.set_ylabel("Approx. KL")
    ax2.set_ylabel("Clip fraction")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=8)
    style_axis(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUTPUT_DIR / "selected_training_metrics.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def save_loss_diagnostics(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df["update"], df["loss"], color="#2563eb", alpha=0.55, label="Loss")
    axes[0].plot(df["update"], rolling(df["loss"]), color="#1e3a8a", linewidth=2, label="Rolling mean (10)")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Combined PPO Loss")
    axes[0].legend()
    style_axis(axes[0])

    axes[1].scatter(
        df["kl"],
        df["clipfrac"],
        c=df["update"],
        cmap="viridis",
        s=30,
        alpha=0.8,
    )
    axes[1].axvline(0.03, color="#dc2626", linestyle="--", linewidth=1)
    axes[1].axhline(0.30, color="#dc2626", linestyle="--", linewidth=1)
    axes[1].set_title("KL vs Clip Fraction")
    axes[1].set_xlabel("Approx. KL")
    axes[1].set_ylabel("Clip fraction")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    path = OUTPUT_DIR / "loss_and_ppo_stability.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(df: pd.DataFrame) -> Path:
    eval_df = df.dropna(subset=["eval_win"])
    best_eval = eval_df.loc[eval_df["eval_win"].idxmax()] if not eval_df.empty else None
    recent = df.tail(min(25, len(df)))
    lines = [
        f"updates={len(df)}",
        f"last_update={int(df['update'].iloc[-1])}",
        f"recent_outcome_mean_25={recent['outcome'].mean():+.4f}",
        f"recent_kl_mean_25={recent['kl'].mean():.5f}",
        f"recent_clipfrac_mean_25={recent['clipfrac'].mean():.4f}",
        f"recent_entropy_mean_25={recent['entropy'].mean():.4f}",
        f"recent_optional_pass_mean_25={recent['optional_pass'].mean():.4f}",
        f"max_kl={df['kl'].max():.5f}",
        f"max_clipfrac={df['clipfrac'].max():.4f}",
    ]
    if best_eval is not None:
        lines.extend(
            [
                f"best_eval_update={int(best_eval['update'])}",
                f"best_eval_win_rate={best_eval['eval_win']:.4f}",
                f"last_eval_win_rate={eval_df['eval_win'].iloc[-1]:.4f}",
            ]
        )
    path = OUTPUT_DIR / "summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    df = parse_history(HISTORY_PATH)
    csv_path = OUTPUT_DIR / "history_parsed.csv"
    df.to_csv(csv_path, index=False)
    paths = [
        save_dashboard(df),
        write_summary(df),
        csv_path,
    ]
    print(f"Parsed {len(df)} updates")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
