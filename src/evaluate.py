"""
evaluate.py — Comprehensive evaluation and metric reporting for SIA.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, brier_score_loss,
)

from src.config import OUTPUTS_DIR, EVAL_REPORT_PATH

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sanity Checks
# ──────────────────────────────────────────────────────────────────────────────

def run_sanity_checks(df: pd.DataFrame) -> list[str]:
    """
    Run automated sanity checks on the output dataframe.
    Returns list of warning messages (empty = all clear).
    """
    warnings = []

    # 1. Pseudo-label class balance
    if "is_mismatch" in df.columns:
        balance = df["is_mismatch"].mean()
        if balance > 0.90:
            warnings.append(f"⚠ Extreme mismatch class imbalance: {100*balance:.1f}% mismatch. Check severity thresholds.")
        if balance < 0.05:
            warnings.append(f"⚠ Very few mismatches detected: {100*balance:.1f}%. Check lexicon coverage.")

    # 2. NaN in predictions
    if "mismatch_probability" in df.columns:
        nan_count = df["mismatch_probability"].isna().sum()
        if nan_count > 0:
            warnings.append(f"⚠ {nan_count} NaN values in mismatch_probability.")

    # 3. Severity confidence
    if "severity_confidence" in df.columns:
        low_conf = (df["severity_confidence"] < 0.40).sum()
        if low_conf / len(df) > 0.30:
            warnings.append(f"⚠ {low_conf:,} tickets ({100*low_conf/len(df):.1f}%) have low severity confidence.")

    # 4. Training set size
    if "include_in_training" in df.columns:
        n_train = df["include_in_training"].sum()
        if n_train < 100:
            warnings.append(f"⚠ Only {n_train} samples in training set. Consider lowering MIN_PSEUDO_LABEL_CONFIDENCE.")

    return warnings


# ──────────────────────────────────────────────────────────────────────────────
# Metric Computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> dict:
    """Compute full classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "auc_roc":          roc_auc_score(y_true, y_prob),
        "f1":               f1_score(y_true, y_pred, zero_division=0),
        "precision":        precision_score(y_true, y_pred, zero_division=0),
        "recall":           recall_score(y_true, y_pred, zero_division=0),
        "brier_score":      brier_score_loss(y_true, y_prob),
        "threshold_used":   threshold,
        "n_positive":       int(y_true.sum()),
        "n_negative":       int((1 - y_true).sum()),
        "class_balance":    float(y_true.mean()),
    }
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Adversarial Subset Analysis
# ──────────────────────────────────────────────────────────────────────────────

def adversarial_analysis(df: pd.DataFrame) -> dict:
    """
    Evaluate performance on the hardest subset:
    - Low-priority tickets (most prone to under-prioritisation)
    - High-priority tickets (most prone to over-prioritisation)
    """
    results = {}
    for priority in ["low", "high", "critical", "medium"]:
        subset = df[df["priority"] == priority]
        if len(subset) > 10 and "is_mismatch" in subset.columns and "mismatch_probability" in subset.columns:
            if subset["is_mismatch"].nunique() > 1:
                results[priority] = {
                    "n":            len(subset),
                    "mismatch_pct": round(100 * subset["is_mismatch"].mean(), 2),
                    "mean_prob":    round(subset["mismatch_probability"].mean(), 4),
                }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_severity_vs_priority(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of assigned priority vs inferred severity."""
    if "inferred_severity" not in df.columns:
        return
    order = ["low", "medium", "high", "critical"]
    ct = pd.crosstab(
        pd.Categorical(df["priority"], categories=order, ordered=True),
        pd.Categorical(df["inferred_severity"], categories=order, ordered=True),
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title("Assigned Priority vs Inferred Severity", fontweight="bold")
    ax.set_xlabel("Inferred Severity")
    ax.set_ylabel("Assigned Priority")
    plt.tight_layout()
    plt.savefig(output_dir / "priority_vs_severity_heatmap.png", dpi=150)
    plt.close()


def plot_mismatch_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of mismatch probability distribution."""
    if "mismatch_probability" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    df["mismatch_probability"].plot.hist(bins=30, edgecolor="white", color="#e74c3c", alpha=0.85, ax=ax)
    ax.set_xlabel("Mismatch Probability")
    ax.set_ylabel("Count")
    ax.set_title("Mismatch Probability Distribution", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "mismatch_probability_distribution.png", dpi=150)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path) -> None:
    """ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Mismatch Classifier", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    """Confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Aligned", "Mismatch"],
                yticklabels=["Aligned", "Mismatch"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Pseudo-Label")
    ax.set_title("Confusion Matrix (on Pseudo-Labels)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Full Evaluation Report
# ──────────────────────────────────────────────────────────────────────────────

def generate_evaluation_report(
    df: pd.DataFrame,
    cv_results: Optional[list] = None,
    oof_probs: Optional[np.ndarray] = None,
    threshold: float = 0.50,
    output_dir: Path = OUTPUTS_DIR,
    verbose: bool = True,
) -> str:
    """
    Generate full evaluation report. Returns report as string.
    Also saves plots and a text report file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 70,
        "  SUPPORT INTEGRITY AUDITOR — EVALUATION REPORT",
        "  Generated: 2026-06-11T22:27:22+05:30",
        "=" * 70,
        "",
    ]

    # ── Dataset overview ──────────────────────────────────────────────────
    lines += [
        "DATASET OVERVIEW",
        "-" * 40,
        f"  Total tickets:        {len(df):>8,}",
    ]
    if "priority" in df.columns:
        lines.append(f"  Priority distribution:")
        for p, c in df["priority"].value_counts().items():
            lines.append(f"    {p:>10s}: {c:>6,}  ({100*c/len(df):.1f}%)")

    lines.append("")

    # ── Severity inference ────────────────────────────────────────────────
    if "inferred_severity" in df.columns:
        lines += [
            "SEVERITY INFERENCE",
            "-" * 40,
            f"  Inferred severity distribution:",
        ]
        for p, c in df["inferred_severity"].value_counts().items():
            lines.append(f"    {p:>10s}: {c:>6,}  ({100*c/len(df):.1f}%)")
        if "severity_confidence" in df.columns:
            lines.append(f"  Mean confidence: {df['severity_confidence'].mean():.4f}")
        lines.append("")

    # ── Pseudo-label stats ────────────────────────────────────────────────
    if "is_mismatch" in df.columns:
        n_mis = df["is_mismatch"].sum()
        lines += [
            "PSEUDO-LABELS",
            "-" * 40,
            f"  Total mismatches:     {n_mis:>8,}  ({100*n_mis/len(df):.1f}%)",
            f"  Aligned:              {len(df)-n_mis:>8,}  ({100*(len(df)-n_mis)/len(df):.1f}%)",
        ]
        if "include_in_training" in df.columns:
            n_train = df["include_in_training"].sum()
            lines.append(f"  In training set:      {n_train:>8,}")
        lines.append("")

    # ── Cross-validation results ──────────────────────────────────────────
    if cv_results:
        lines += ["CROSS-VALIDATION (on pseudo-labels)", "-" * 40]

        aucs = [r["auc"] for r in cv_results]
        f1s  = [r["f1"] for r in cv_results]
        lines.append(f"  {'Fold':>5}  {'AUC':>8}  {'F1':>8}  {'Prec':>8}  {'Rec':>8}")
        lines.append(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for r in cv_results:
            lines.append(
                f"  {r['fold']:>5}  {r['auc']:>8.4f}  {r['f1']:>8.4f}  "
                f"{r['precision']:>8.4f}  {r['recall']:>8.4f}"
            )
        lines.append(f"  {'Mean':>5}  {np.mean(aucs):>8.4f}  {np.mean(f1s):>8.4f}")
        lines.append(f"  {'Std':>5}  {np.std(aucs):>8.4f}  {np.std(f1s):>8.4f}")
        lines.append("")

    # ── OOF metrics ───────────────────────────────────────────────────────
    if oof_probs is not None and "is_mismatch" in df:
        train_mask = df.get("include_in_training", pd.Series(True, index=df.index))
        y_true = df.loc[train_mask, "is_mismatch"].values
        if len(oof_probs) == len(y_true):
            metrics = compute_metrics(y_true, oof_probs, threshold)
            lines += [
                "OOF METRICS",
                "-" * 40,
                f"  AUC-ROC:       {metrics['auc_roc']:.4f}",
                f"  F1 Score:      {metrics['f1']:.4f}",
                f"  Precision:     {metrics['precision']:.4f}",
                f"  Recall:        {metrics['recall']:.4f}",
                f"  Brier Score:   {metrics['brier_score']:.4f}",
                f"  Threshold:     {threshold:.4f}",
                "",
                "  Classification Report (OOF):",
            ]
            y_pred = (oof_probs >= threshold).astype(int)
            cr = classification_report(y_true, y_pred, target_names=["Aligned", "Mismatch"], zero_division=0)
            lines += [f"  {l}" for l in cr.split("\n")]
            lines.append("")

            # Plots
            try:
                plot_roc_curve(y_true, oof_probs, output_dir)
                plot_confusion_matrix(y_true, y_pred, output_dir)
                lines.append(f"  Plots saved to {output_dir}")
            except Exception as e:
                logger.warning(f"Plot generation failed: {e}")

    # ── Prediction distribution ───────────────────────────────────────────
    if "mismatch_predicted" in df.columns:
        n_pred_mis = df["mismatch_predicted"].sum()
        lines += [
            "PREDICTIONS",
            "-" * 40,
            f"  Predicted mismatches: {n_pred_mis:>8,}  ({100*n_pred_mis/len(df):.1f}%)",
            "",
        ]
        try:
            plot_mismatch_distribution(df, output_dir)
            plot_severity_vs_priority(df, output_dir)
        except Exception as e:
            logger.warning(f"Distribution plot failed: {e}")

    # ── Adversarial analysis ──────────────────────────────────────────────
    if "mismatch_probability" in df.columns:
        adv = adversarial_analysis(df)
        if adv:
            lines += ["ADVERSARIAL ANALYSIS (by assigned priority)", "-" * 40]
            for prio, stats in adv.items():
                lines.append(f"  [{prio.upper():>8s}]  n={stats['n']:>5,}  "
                              f"mismatch={stats['mismatch_pct']:.1f}%  "
                              f"mean_prob={stats['mean_prob']:.4f}")
            lines.append("")

    # ── Sanity checks ─────────────────────────────────────────────────────
    warnings = run_sanity_checks(df)
    lines += ["SANITY CHECKS", "-" * 40]
    if warnings:
        lines += [f"  {w}" for w in warnings]
    else:
        lines.append("  ✓ All sanity checks passed.")
    lines += ["", "=" * 70]

    report = "\n".join(lines)

    with open(EVAL_REPORT_PATH, "w") as f:
        f.write(report)

    if verbose:
        print(report)
        print(f"\n[Evaluate] Report saved to {EVAL_REPORT_PATH}")

    return report
