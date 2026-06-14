"""
data_loader.py — Load, validate, and normalize the CRM tickets dataset.
Handles column name variations, missing values, and basic preprocessing.
"""

import re
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.config import COLUMN_ALIASES, PRIORITY_LABEL_MAP

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Column Resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_columns(df: pd.DataFrame) -> dict:
    """
    Map standardized internal column names → actual DataFrame column names.
    Returns a dict: internal_name → actual_col_name (or None if missing).
    """
    resolved = {}
    available = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for internal, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            key = alias.lower().strip().replace(" ", "_")
            if key in available:
                found = available[key]
                break
        resolved[internal] = found
    return resolved


def rename_to_standard(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Rename resolved columns to standardized internal names."""
    rename = {v: k for k, v in col_map.items() if v is not None}
    df = df.rename(columns=rename)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Priority Normalisation
# ──────────────────────────────────────────────────────────────────────────────

_PRIORITY_PATTERNS = {
    "critical": r"\b(critical|urgent|emergency|p0|p1|sev1|severity[\s\-]?1)\b",
    "high":     r"\b(high|hi|p2|sev2|severity[\s\-]?2)\b",
    "medium":   r"\b(medium|med|moderate|p3|sev3|severity[\s\-]?3)\b",
    "low":      r"\b(low|lo|minor|p4|p5|sev4|sev5|severity[\s\-]?[45])\b",
}


def normalise_priority(raw: str) -> Optional[str]:
    """Map any priority string to one of: low / medium / high / critical."""
    if pd.isna(raw):
        return None
    s = str(raw).lower().strip()
    for label, pattern in _PRIORITY_PATTERNS.items():
        if re.search(pattern, s):
            return label
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Light cleaning: collapse whitespace, decode HTML entities, normalise quotes."""
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Structural Feature Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def parse_time_column(series: pd.Series) -> pd.Series:
    """
    Convert time columns to numeric (minutes).
    Handles formats: '2 hours 30 minutes', '1.5', '90 min', float/int seconds.
    """
    def _to_minutes(val):
        if pd.isna(val):
            return np.nan
        s = str(val).lower().strip()
        try:
            return float(s) / 60.0  # assume seconds if plain number
        except ValueError:
            pass
        total = 0.0
        for m, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(day|hour|hr|minute|min|second|sec)", s):
            v = float(m)
            if "day" in unit:     total += v * 1440
            elif "hour" in unit or "hr" == unit: total += v * 60
            elif "min" in unit:   total += v
            elif "sec" in unit:   total += v / 60
        return total if total > 0 else np.nan
    return series.apply(_to_minutes)


# ──────────────────────────────────────────────────────────────────────────────
# Main Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str | Path, verbose: bool = True) -> pd.DataFrame:
    """
    Load the CRM support ticket CSV, normalise columns, clean text,
    and return a tidy DataFrame ready for downstream stages.

    Parameters
    ----------
    filepath : path to the CSV file
    verbose  : print column resolution info

    Returns
    -------
    pd.DataFrame with standardised columns + derived text fields
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            f"Please place your CSV in the data/ folder."
        )

    logger.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    original_cols = list(df.columns)
    logger.info(f"  → {len(df):,} rows, {len(df.columns)} columns")

    # ── Resolve & rename columns ──────────────────────────────────────────
    col_map = resolve_columns(df)
    if verbose:
        print("\n[DataLoader] Column Resolution:")
        for internal, actual in col_map.items():
            status = f"✓ '{actual}'" if actual else "✗ NOT FOUND (will be None)"
            print(f"  {internal:20s} → {status}")

    df = rename_to_standard(df, col_map)

    # ── Ensure required columns exist ────────────────────────────────────
    required = ["subject", "description", "priority"]
    missing = [c for c in required if c not in df.columns or df[c].isna().all()]
    if missing:
        raise ValueError(
            f"Required columns missing or entirely empty: {missing}\n"
            f"Original columns found: {original_cols}"
        )

    # ── Priority normalisation ────────────────────────────────────────────
    df["priority_raw"] = df["priority"].copy()
    df["priority"] = df["priority"].apply(normalise_priority)

    pre_drop = len(df)
    df = df[df["priority"].notna()].copy()
    post_drop = len(df)
    if verbose and pre_drop != post_drop:
        print(f"  Dropped {pre_drop - post_drop} rows with unrecognisable priority")

    df["priority_rank"] = df["priority"].map(PRIORITY_LABEL_MAP)

    # ── Text cleaning ─────────────────────────────────────────────────────
    df["subject"]     = df["subject"].apply(clean_text)
    df["description"] = df["description"].apply(clean_text)

    # Combine subject + description as the primary NLP input
    df["full_text"] = (df["subject"] + " [SEP] " + df["description"]).str.strip()

    if "resolution" in df.columns:
        df["resolution"] = df["resolution"].apply(clean_text)

    # ── Time columns ──────────────────────────────────────────────────────
    for col in ["first_response", "resolution_time"]:
        if col in df.columns:
            df[f"{col}_minutes"] = parse_time_column(df[col])

    # ── CSAT ──────────────────────────────────────────────────────────────
    if "csat" in df.columns:
        df["csat_numeric"] = pd.to_numeric(df["csat"], errors="coerce")

    # ── Ticket ID ─────────────────────────────────────────────────────────
    if "ticket_id" not in df.columns:
        df["ticket_id"] = [f"TKT-{i:06d}" for i in range(len(df))]
    df["ticket_id"] = df["ticket_id"].astype(str).str.strip()
    df = df.reset_index(drop=True)

    logger.info(f"  → After cleaning: {len(df):,} rows ready")
    if verbose:
        print(f"\n[DataLoader] Priority distribution:")
        print(df["priority"].value_counts().to_string())
        print(f"\n[DataLoader] Final shape: {df.shape}")

    return df


def get_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-row text statistics used as structural features."""
    stats = pd.DataFrame(index=df.index)
    stats["text_length"]       = df["full_text"].str.len()
    stats["word_count"]        = df["full_text"].str.split().str.len()
    stats["exclamation_count"] = df["full_text"].str.count(r"!")
    stats["question_count"]    = df["full_text"].str.count(r"\?")
    stats["caps_ratio"]        = df["full_text"].apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )
    stats["digit_count"]       = df["full_text"].str.count(r"\d")
    stats["url_count"]         = df["full_text"].str.count(
        r"https?://\S+|www\.\S+"
    )
    return stats
