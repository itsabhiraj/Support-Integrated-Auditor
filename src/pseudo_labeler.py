"""
pseudo_labeler.py — Stage 2: Generate pseudo-labels for mismatch detection.

Core logic:
  - Compare inferred severity rank vs assigned priority rank
  - If |gap| >= MISMATCH_GAP_THRESHOLD → is_mismatch = 1
  - Filter by confidence to ensure label quality
  - Output pseudo_labels.csv with confidence scores
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    MISMATCH_GAP_THRESHOLD,
    MIN_PSEUDO_LABEL_CONFIDENCE,
    PRIORITY_LABEL_MAP,
    PSEUDO_LABELS_PATH,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Mismatch Direction Labels
# ──────────────────────────────────────────────────────────────────────────────

def _mismatch_direction(gap: int) -> str:
    """Human-readable mismatch direction."""
    if gap < 0:
        return "OVER-PRIORITISED"   # Agent assigned higher priority than warranted
    elif gap > 0:
        return "UNDER-PRIORITISED"  # Agent assigned lower priority than warranted
    return "ALIGNED"


def _pseudo_label_confidence(
    gap: int,
    severity_confidence: float,
    severity_score: float,
    priority_rank: int,
) -> float:
    """
    Confidence for the pseudo-label:
    - Increases with larger gap (clear mismatch)
    - Increases with higher severity model confidence
    - Slightly penalised for boundary scores (near threshold)
    """
    # Base: severity model confidence
    base = severity_confidence

    # Boost for larger absolute gap
    gap_boost = min(abs(gap) * 0.15, 0.30)

    # Penalise near-threshold scores
    boundaries = [0.30, 0.55, 0.78]
    min_dist_to_boundary = min(abs(severity_score - b) for b in boundaries)
    boundary_penalty = 0.10 if min_dist_to_boundary < 0.05 else 0.0

    confidence = base + gap_boost - boundary_penalty
    return float(np.clip(confidence, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Main Pseudo-Labeler
# ──────────────────────────────────────────────────────────────────────────────

def generate_pseudo_labels(
    df: pd.DataFrame,
    gap_threshold: int = MISMATCH_GAP_THRESHOLD,
    min_confidence: float = MIN_PSEUDO_LABEL_CONFIDENCE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate pseudo-labels for every ticket.

    Required input columns (from severity_inferrer output):
      - priority_rank
      - inferred_severity_rank
      - inferred_severity_score
      - severity_confidence

    Adds columns:
      - priority_gap           : inferred_rank - priority_rank
      - is_mismatch            : binary {0, 1}
      - mismatch_direction     : OVER-PRIORITISED / UNDER-PRIORITISED / ALIGNED
      - pseudo_label_confidence: float [0, 1]
      - pseudo_label_quality   : HIGH / MEDIUM / LOW / UNCERTAIN
      - include_in_training    : bool (confidence >= threshold)
    """
    df = df.copy()

    required = ["priority_rank", "inferred_severity_rank", "inferred_severity_score", "severity_confidence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns from severity inference: {missing}. Run severity_inferrer first.")

    # ── Ordinal gap ──────────────────────────────────────────────────────
    df["priority_gap"] = (
        df["inferred_severity_rank"].astype(int) - df["priority_rank"].astype(int)
    )

    # ── Binary mismatch label ────────────────────────────────────────────
    df["is_mismatch"] = (df["priority_gap"].abs() >= gap_threshold).astype(int)
    df["mismatch_direction"] = df["priority_gap"].apply(_mismatch_direction)

    # ── Confidence scoring ────────────────────────────────────────────────
    df["pseudo_label_confidence"] = [
        _pseudo_label_confidence(
            int(df["priority_gap"].iloc[i]),
            float(df["severity_confidence"].iloc[i]),
            float(df["inferred_severity_score"].iloc[i]),
            int(df["priority_rank"].iloc[i]),
        )
        for i in range(len(df))
    ]

    # ── Quality tier ──────────────────────────────────────────────────────
    def _quality_tier(conf):
        if conf >= 0.75:   return "HIGH"
        elif conf >= 0.55: return "MEDIUM"
        elif conf >= 0.40: return "LOW"
        return "UNCERTAIN"

    df["pseudo_label_quality"] = df["pseudo_label_confidence"].apply(_quality_tier)

    # ── Training inclusion ────────────────────────────────────────────────
    df["include_in_training"] = (df["pseudo_label_confidence"] >= min_confidence)

    if verbose:
        n_total    = len(df)
        n_mismatch = df["is_mismatch"].sum()
        n_train    = df["include_in_training"].sum()
        n_excluded = n_total - n_train

        print(f"\n[PseudoLabeler] Results:")
        print(f"  Total tickets:        {n_total:>6,}")
        print(f"  Mismatches flagged:   {n_mismatch:>6,}  ({100*n_mismatch/n_total:.1f}%)")
        print(f"  Aligned (OK):         {n_total-n_mismatch:>6,}  ({100*(n_total-n_mismatch)/n_total:.1f}%)")
        print(f"  Included in training: {n_train:>6,}  (conf ≥ {min_confidence})")
        print(f"  Excluded (uncertain): {n_excluded:>6,}")

        print(f"\n  Mismatch direction:")
        print(df["mismatch_direction"].value_counts().to_string())

        print(f"\n  Pseudo-label quality:")
        print(df["pseudo_label_quality"].value_counts().to_string())

        # Class balance check
        train_df = df[df["include_in_training"]]
        if len(train_df) > 0:
            balance = train_df["is_mismatch"].mean()
            print(f"\n  Training class balance: {100*balance:.1f}% mismatch")
            if balance > 0.90 or balance < 0.10:
                logger.warning("⚠ Extreme class imbalance in pseudo-labels. Check severity thresholds.")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────────────────────────────────────

PSEUDO_LABEL_COLS = [
    "ticket_id", "priority", "priority_rank",
    "inferred_severity", "inferred_severity_rank", "inferred_severity_score",
    "urgency_lexicon_score", "nli_severity_score", "structural_score", "sentiment_score",
    "severity_confidence",
    "priority_gap", "is_mismatch", "mismatch_direction",
    "pseudo_label_confidence", "pseudo_label_quality", "include_in_training",
    "full_text", "subject", "description",
]


def save_pseudo_labels(df: pd.DataFrame, path: Path = PSEUDO_LABELS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Only save columns that exist
    cols = [c for c in PSEUDO_LABEL_COLS if c in df.columns]
    df[cols].to_csv(path, index=False)
    logger.info(f"Pseudo-labels saved to {path}  ({len(df):,} rows)")


def load_pseudo_labels(path: Path = PSEUDO_LABELS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pseudo-labels not found at {path}. Run --stage infer first.")
    return pd.read_csv(path)


def get_training_set(df: pd.DataFrame) -> pd.DataFrame:
    """Return the confident subset for classifier training."""
    train = df[df["include_in_training"]].copy()
    logger.info(f"Training set: {len(train):,} rows, "
                f"{train['is_mismatch'].mean()*100:.1f}% mismatch")
    return train
