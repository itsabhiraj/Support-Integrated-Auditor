"""
severity_inferrer.py — Stage 1: Infer objective ticket severity.

Ensemble of four independent signals:
  1. Urgency Lexicon Score      (rule-based, fast)
  2. NLI Zero-Shot Score        (facebook/bart-large-mnli, --offline disables)
  3. Structural Signal Score    (CSAT, resolution time, response delay)
  4. Sentiment Intensity Score  (VADER)

Final `inferred_severity_score` ∈ [0, 1] → mapped to Low/Medium/High/Critical
"""

import re
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    CRITICAL_SIGNALS, HIGH_SIGNALS, MEDIUM_SIGNALS, LOW_SIGNALS,
    NEGATION_WORDS, NLI_MODEL_NAME, NLI_BATCH_SIZE, NLI_SEVERITY_LABELS,
    SEVERITY_WEIGHTS, SEVERITY_THRESHOLDS, PRIORITY_LABEL_MAP,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Urgency Lexicon Score
# ──────────────────────────────────────────────────────────────────────────────

def _compile_lexicon_patterns() -> dict:
    """Pre-compile regex patterns for speed."""
    return {
        "critical": [re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE) for p in CRITICAL_SIGNALS],
        "high":     [re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE) for p in HIGH_SIGNALS],
        "medium":   [re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE) for p in MEDIUM_SIGNALS],
        "low":      [re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE) for p in LOW_SIGNALS],
        "negation": [re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE) for p in NEGATION_WORDS],
    }


# Module-level compiled patterns (compiled once)
_LEXICON_PATTERNS = _compile_lexicon_patterns()
_LEVEL_SCORES = {"critical": 3, "high": 2, "medium": 1, "low": 0}


def _score_lexicon(text: str) -> tuple[float, dict]:
    """
    Compute lexicon hit scores. Returns normalised score [0,1] and hit details.
    Negation within a 3-word window of a match reduces its weight by 50%.
    """
    words = text.lower().split()
    negation_positions = set()
    for i, w in enumerate(words):
        if any(p.search(w) for p in _LEXICON_PATTERNS["negation"]):
            negation_positions.update(range(max(0, i - 1), min(len(words), i + 4)))

    hits = {level: 0.0 for level in ["critical", "high", "medium", "low"]}
    evidence = {level: [] for level in hits}

    for level, patterns in _LEXICON_PATTERNS.items():
        if level == "negation":
            continue
        for pat in patterns:
            for m in pat.finditer(text):
                span_text = m.group()
                char_pos = m.start()
                word_pos = len(text[:char_pos].split())
                weight = 0.5 if word_pos in negation_positions else 1.0
                hits[level] += weight
                evidence[level].append(span_text)

    # Weighted sum normalised to [0, 1]
    total_hits = sum(hits.values()) + 1e-9
    weighted = sum(_LEVEL_SCORES[l] * hits[l] for l in hits)
    max_possible = 3.0 * total_hits
    score = weighted / max_possible if max_possible > 0 else 0.0
    score = float(np.clip(score, 0.0, 1.0))
    return score, evidence


# ──────────────────────────────────────────────────────────────────────────────
# 2. NLI Zero-Shot Score
# ──────────────────────────────────────────────────────────────────────────────

def _load_nli_pipeline():
    """Lazy-load the HuggingFace zero-shot pipeline."""
    from transformers import pipeline
    logger.info(f"Loading NLI model: {NLI_MODEL_NAME} (this may take a moment on first run)")
    pipe = pipeline(
        "zero-shot-classification",
        model=NLI_MODEL_NAME,
        device=-1,           # CPU; set to 0 for GPU
        batch_size=NLI_BATCH_SIZE,
    )
    return pipe


def _score_nli_batch(texts: list[str], pipe) -> list[float]:
    """
    Use zero-shot NLI to score severity of a batch of texts.
    Returns list of [0, 1] normalised scores.
    """
    # Truncate texts for model input limit
    texts_trunc = [t[:512] for t in texts]
    results = pipe(
        texts_trunc,
        candidate_labels=NLI_SEVERITY_LABELS,
        multi_label=False,
    )
    scores_out = []
    # Label ordering: [critical=1.0, high=0.67, medium=0.33, low=0.0]
    label_to_score = {
        NLI_SEVERITY_LABELS[0]: 1.0,   # critical
        NLI_SEVERITY_LABELS[1]: 0.67,  # high
        NLI_SEVERITY_LABELS[2]: 0.33,  # medium
        NLI_SEVERITY_LABELS[3]: 0.0,   # low
    }
    for res in results:
        # Weighted sum by NLI probability
        score = sum(label_to_score[l] * s for l, s in zip(res["labels"], res["scores"]))
        scores_out.append(float(score))
    return scores_out


# ──────────────────────────────────────────────────────────────────────────────
# 3. Structural Signal Score
# ──────────────────────────────────────────────────────────────────────────────

def _score_structural(df: pd.DataFrame) -> np.ndarray:
    """
    Build structural severity signal ∈ [0, 1]:
    - Low CSAT → higher severity signal
    - Long resolution time → higher severity signal
    - Long response time → higher severity signal
    """
    scores = np.zeros(len(df), dtype=np.float32)
    weight_total = 0.0

    if "csat_numeric" in df.columns and df["csat_numeric"].notna().sum() > 10:
        csat = df["csat_numeric"].fillna(df["csat_numeric"].median())
        lo, hi = csat.min(), csat.max()
        if hi > lo:
            csat_signal = 1.0 - (csat - lo) / (hi - lo)  # invert: low CSAT = high severity
        else:
            csat_signal = pd.Series(0.5, index=df.index)
        scores += 0.5 * csat_signal.values
        weight_total += 0.5

    if "resolution_time_minutes" in df.columns and df["resolution_time_minutes"].notna().sum() > 10:
        rt = df["resolution_time_minutes"].fillna(0)
        rt_log = np.log1p(rt.values)
        rt_norm = (rt_log - rt_log.min()) / (rt_log.max() - rt_log.min() + 1e-9)
        scores += 0.35 * rt_norm
        weight_total += 0.35

    if "first_response_minutes" in df.columns and df["first_response_minutes"].notna().sum() > 10:
        fr = df["first_response_minutes"].fillna(0)
        fr_log = np.log1p(fr.values)
        fr_norm = (fr_log - fr_log.min()) / (fr_log.max() - fr_log.min() + 1e-9)
        scores += 0.15 * fr_norm
        weight_total += 0.15

    if weight_total > 0:
        scores = scores / weight_total
    else:
        scores = np.full(len(df), 0.5, dtype=np.float32)

    return np.clip(scores, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Sentiment Intensity (VADER)
# ──────────────────────────────────────────────────────────────────────────────

def _load_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def _score_sentiment(texts: pd.Series, analyzer) -> np.ndarray:
    """
    VADER compound score → severity signal.
    Negative sentiment → higher severity. Neutral-to-positive → lower.
    """
    compounds = texts.apply(lambda t: analyzer.polarity_scores(str(t))["compound"])
    # compound ∈ [-1, 1]; map to [0, 1] severity signal (invert + normalise)
    severity_signal = (1.0 - (compounds + 1.0) / 2.0)  # negative → near 1
    # Add intensity from exclamation / CAPS markers
    exclaim_boost = texts.str.count(r"!").clip(upper=5) / 10.0
    caps_boost = texts.apply(
        lambda t: min(sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1) * 3, 0.2)
    )
    score = severity_signal + exclaim_boost + caps_boost
    return np.clip(score.values, 0.0, 1.0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble & Thresholding
# ──────────────────────────────────────────────────────────────────────────────

def _score_to_label(score: float) -> str:
    for label, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= score < hi:
            return label
    return "critical"  # fallback for score = 1.0


def _compute_confidence(
    lexicon_score: float,
    nli_score: float,
    use_nli: bool,
    structural_score: float,
    sentiment_score: float,
) -> float:
    """
    Confidence = how much the signals agree with each other.
    High confidence when lexicon and NLI scores are close.
    """
    signals = [lexicon_score, structural_score, sentiment_score]
    if use_nli:
        signals.append(nli_score)
    std = float(np.std(signals))
    # Low std → high agreement → high confidence
    confidence = float(np.clip(1.0 - 2.0 * std, 0.0, 1.0))
    return confidence


# ──────────────────────────────────────────────────────────────────────────────
# Main Public API
# ──────────────────────────────────────────────────────────────────────────────

def infer_severity(
    df: pd.DataFrame,
    offline: bool = False,
    nli_pipe=None,
    vader=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run severity inference on the full dataframe.
    Adds columns:
      - urgency_lexicon_score
      - nli_severity_score   (0 if offline)
      - structural_score
      - sentiment_score
      - inferred_severity_score  (ensemble)
      - inferred_severity        (Low/Medium/High/Critical label)
      - severity_confidence      (0–1)
      - lexicon_evidence         (dict of matched terms)

    Parameters
    ----------
    df       : output of data_loader.load_data()
    offline  : if True, skip NLI model (use lexicon+structural+sentiment only)
    nli_pipe : pre-loaded NLI pipeline (optional, to avoid re-loading)
    vader    : pre-loaded VADER analyzer (optional)
    """
    df = df.copy()
    N = len(df)
    texts = df["full_text"]

    if verbose:
        print(f"\n[SeverityInferrer] Processing {N:,} tickets...")

    # ── Load models (lazy) ─────────────────────────────────────────────
    if vader is None:
        try:
            vader = _load_vader()
        except Exception as e:
            logger.warning(f"VADER failed to load ({e}); sentiment score set to 0.5")
            vader = None

    if not offline and nli_pipe is None:
        try:
            nli_pipe = _load_nli_pipeline()
        except Exception as e:
            logger.warning(f"NLI model failed ({e}); falling back to offline mode")
            nli_pipe = None
            offline = True

    # ── Signal 1: Lexicon ──────────────────────────────────────────────
    if verbose:
        print("  → Running lexicon scoring...")
    lex_scores = []
    lex_evidences = []
    for text in texts:
        score, evidence = _score_lexicon(str(text))
        lex_scores.append(score)
        lex_evidences.append(evidence)

    df["urgency_lexicon_score"] = lex_scores
    df["lexicon_evidence"] = lex_evidences

    # ── Signal 2: NLI ─────────────────────────────────────────────────
    if not offline and nli_pipe is not None:
        if verbose:
            print("  → Running NLI zero-shot scoring...")
        from tqdm import tqdm
        nli_scores = []
        batch_size = NLI_BATCH_SIZE
        texts_list = texts.tolist()
        for i in tqdm(range(0, N, batch_size), desc="NLI", unit="batch", disable=not verbose):
            batch = texts_list[i:i + batch_size]
            nli_scores.extend(_score_nli_batch(batch, nli_pipe))
        df["nli_severity_score"] = nli_scores
        use_nli = True
    else:
        df["nli_severity_score"] = 0.5
        use_nli = False

    # ── Signal 3: Structural ──────────────────────────────────────────
    if verbose:
        print("  → Running structural scoring...")
    df["structural_score"] = _score_structural(df)

    # ── Signal 4: Sentiment ───────────────────────────────────────────
    if vader is not None:
        if verbose:
            print("  → Running sentiment scoring...")
        df["sentiment_score"] = _score_sentiment(texts, vader)
    else:
        df["sentiment_score"] = 0.5

    # ── Ensemble ──────────────────────────────────────────────────────
    weights = SEVERITY_WEIGHTS.copy()
    if not use_nli:
        # Redistribute NLI weight to lexicon
        weights["lexicon"] += weights["nli"]
        weights["nli"] = 0.0

    df["inferred_severity_score"] = (
        weights["lexicon"]    * df["urgency_lexicon_score"] +
        weights["nli"]        * df["nli_severity_score"] +
        weights["structural"] * df["structural_score"] +
        weights["sentiment"]  * df["sentiment_score"]
    ).clip(0.0, 1.0)

    # ── Label assignment ──────────────────────────────────────────────
    df["inferred_severity"] = df["inferred_severity_score"].apply(_score_to_label)
    df["inferred_severity_rank"] = df["inferred_severity"].map(PRIORITY_LABEL_MAP)

    # ── Confidence ────────────────────────────────────────────────────
    df["severity_confidence"] = [
        _compute_confidence(
            lex_scores[i],
            float(df["nli_severity_score"].iloc[i]),
            use_nli,
            float(df["structural_score"].iloc[i]),
            float(df["sentiment_score"].iloc[i]),
        )
        for i in range(N)
    ]

    if verbose:
        print(f"\n[SeverityInferrer] Inferred severity distribution:")
        print(df["inferred_severity"].value_counts().to_string())
        print(f"  Mean confidence: {df['severity_confidence'].mean():.3f}")

    return df
