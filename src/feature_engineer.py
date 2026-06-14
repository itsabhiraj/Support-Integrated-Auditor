"""
feature_engineer.py — Build the full mixed feature matrix for the mismatch classifier.

Features:
  1. TF-IDF on full_text (char+word grams)
  2. SBERT sentence embeddings (all-MiniLM-L6-v2)
  3. Structural / metadata features
  4. Derived severity signal features (from severity_inferrer)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, RobustScaler

from src.config import (
    SBERT_MODEL_NAME,
    TFIDF_MAX_FEATURES,
    MODELS_DIR,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF Vectorizer
# ──────────────────────────────────────────────────────────────────────────────

def build_tfidf(
    texts: pd.Series,
    fit: bool = True,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> tuple[csr_matrix, TfidfVectorizer]:
    """
    Build or reuse a TF-IDF vectorizer.
    Uses both word and character n-grams for robustness.
    """
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            analyzer="word",
            sublinear_tf=True,
            min_df=2,
            dtype=np.float32,
        )
        X = vectorizer.fit_transform(texts.fillna(""))
        logger.info(f"TF-IDF fitted: {X.shape}")
    else:
        if vectorizer is None:
            raise ValueError("vectorizer must be provided when fit=False")
        X = vectorizer.transform(texts.fillna(""))

    return X, vectorizer


# ──────────────────────────────────────────────────────────────────────────────
# SBERT Embeddings
# ──────────────────────────────────────────────────────────────────────────────

def build_sbert_embeddings(
    texts: pd.Series,
    model=None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode texts with Sentence-BERT (all-MiniLM-L6-v2).
    Returns a (N, 384) float32 array.
    """
    from sentence_transformers import SentenceTransformer

    if model is None:
        logger.info(f"Loading SBERT model: {SBERT_MODEL_NAME}")
        model = SentenceTransformer(SBERT_MODEL_NAME)

    texts_list = texts.fillna("").tolist()
    embeddings = model.encode(
        texts_list,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise for cosine-friendly downstream use
    )
    logger.info(f"SBERT embeddings: {embeddings.shape}")
    return embeddings.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Structural / Metadata Features
# ──────────────────────────────────────────────────────────────────────────────

def build_structural_features(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: Optional[dict] = None,
    scaler: Optional[RobustScaler] = None,
) -> tuple[np.ndarray, dict, RobustScaler]:
    """
    Build structural feature matrix from metadata columns.

    Returns
    -------
    X_struct   : (N, F) float32 array
    encoders   : dict of LabelEncoders
    scaler     : fitted RobustScaler
    """
    rows = pd.DataFrame(index=df.index)

    # ── Text stats ──────────────────────────────────────────────────────
    rows["text_length"]       = df["full_text"].str.len().fillna(0)
    rows["word_count"]        = df["full_text"].str.split().str.len().fillna(0)
    rows["exclamation_count"] = df["full_text"].str.count(r"!").fillna(0)
    rows["question_count"]    = df["full_text"].str.count(r"\?").fillna(0)
    rows["caps_ratio"]        = df["full_text"].apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
    )

    # ── CSAT (inverted — low satisfaction → potentially high severity) ──
    if "csat_numeric" in df.columns:
        csat = df["csat_numeric"].fillna(df["csat_numeric"].median())
        rows["csat_norm"] = (csat - csat.min()) / (csat.max() - csat.min() + 1e-9)
        rows["csat_inverted"] = 1.0 - rows["csat_norm"]
    else:
        rows["csat_norm"] = 0.5
        rows["csat_inverted"] = 0.5

    # ── Resolution & Response time (normalised) ──────────────────────────
    for col in ["first_response_minutes", "resolution_time_minutes"]:
        if col in df.columns:
            vals = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
            rows[f"{col}_norm"] = np.log1p(vals)
        else:
            rows[f"{col}_norm"] = 0.0

    # ── Categorical encodings ────────────────────────────────────────────
    if encoders is None:
        encoders = {}

    for col in ["channel", "ticket_type", "status"]:
        if col in df.columns:
            series = df[col].fillna("unknown").astype(str).str.lower().str.strip()
            if fit:
                le = LabelEncoder()
                rows[f"{col}_enc"] = le.fit_transform(series)
                encoders[col] = le
            else:
                le = encoders.get(col)
                if le is not None:
                    known = set(le.classes_)
                    series = series.apply(lambda x: x if x in known else "unknown")
                    if "unknown" not in le.classes_:
                        # fallback: encode unseen as 0
                        rows[f"{col}_enc"] = series.apply(
                            lambda x: le.transform([x])[0] if x in known else 0
                        )
                    else:
                        rows[f"{col}_enc"] = le.transform(series)
                else:
                    rows[f"{col}_enc"] = 0
        else:
            rows[f"{col}_enc"] = 0

    # ── Severity-derived signals (injected later by severity_inferrer) ──
    for col in ["inferred_severity_score", "severity_confidence", "urgency_lexicon_score"]:
        rows[col] = df[col].fillna(0.0) if col in df.columns else 0.0

    X = rows.values.astype(np.float32)

    # ── Scale ─────────────────────────────────────────────────────────────
    if fit:
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
    else:
        if scaler is not None:
            X = scaler.transform(X)

    logger.info(f"Structural features: {X.shape}")
    return X, encoders, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Combined Feature Assembly
# ──────────────────────────────────────────────────────────────────────────────

def assemble_features(
    X_tfidf: csr_matrix,
    X_sbert: np.ndarray,
    X_struct: np.ndarray,
) -> np.ndarray:
    """
    Concatenate TF-IDF (sparse → dense via truncation), SBERT, and structural.
    Returns dense (N, D) float32 array.
    """
    # Convert sparse TF-IDF to dense for LightGBM
    tfidf_dense = X_tfidf.toarray().astype(np.float32)
    X = np.concatenate([tfidf_dense, X_sbert, X_struct], axis=1)
    logger.info(f"Assembled feature matrix: {X.shape}")
    return X


def get_feature_names(
    vectorizer: TfidfVectorizer,
    sbert_dim: int,
    struct_cols: list,
) -> list:
    """Return human-readable feature names for SHAP interpretation."""
    tfidf_names = [f"tfidf:{v}" for v in vectorizer.get_feature_names_out()]
    sbert_names = [f"sbert_{i}" for i in range(sbert_dim)]
    return tfidf_names + sbert_names + struct_cols


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_feature_artifacts(
    vectorizer: TfidfVectorizer,
    encoders: dict,
    scaler: RobustScaler,
    sbert_model=None,
    path: Path = MODELS_DIR,
):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(path / "label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(path / "robust_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Feature artifacts saved to {path}")


def load_feature_artifacts(path: Path = MODELS_DIR) -> tuple:
    path = Path(path)
    with open(path / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(path / "label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open(path / "robust_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return vectorizer, encoders, scaler


# ────────────────────────────────────────────────────────────────────────────
# High-level feature builder
# ────────────────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    fit: bool = True,
    vectorizer: Optional[TfidfVectorizer] = None,
    encoders: Optional[dict] = None,
    scaler: Optional[RobustScaler] = None,
    sbert_model=None,
    severity_offline: bool = True,
    save_artifacts_flag: bool = False,
    artifacts_path: Path = MODELS_DIR,
    verbose: bool = True,
) -> tuple[np.ndarray, csr_matrix, TfidfVectorizer, dict, RobustScaler, list]:
    """
    Build full feature matrix for a dataframe.

    Steps:
      1. Ensure `full_text` exists (fall back to `subject`+`description`).
      2. Run `infer_severity` to add severity-derived signals.
      3. Fit / transform TF-IDF on `full_text`.
      4. Encode SBERT embeddings for `full_text`.
      5. Build structural features (categoricals, text stats, severity signals).
      6. Assemble and return dense feature matrix and artifacts.

    Returns
    -------
    X          : dense feature matrix (N, D)
    vectorizer : fitted TfidfVectorizer
    encoders   : label encoders dict
    scaler     : fitted RobustScaler
    feature_names : list of feature names
    """
    from src.severity_inferrer import infer_severity

    df = df.copy()

    # Ensure full_text exists
    if "full_text" not in df.columns:
        parts = []
        if "subject" in df.columns:
            parts.append(df["subject"].fillna(""))
        if "description" in df.columns:
            parts.append(df["description"].fillna(""))
        if parts:
            df["full_text"] = (" \n ").join(parts)
        else:
            df["full_text"] = ""

    # 1) Severity inference (injects inferred_severity_score, severity_confidence, etc.)
    try:
        if verbose:
            logger.info("Running severity inference...")
        df = infer_severity(df, offline=severity_offline, verbose=verbose)
    except Exception as e:
        logger.warning(f"Severity inference failed ({e}); continuing without it")

    # 2) TF-IDF
    if verbose:
        logger.info("Building TF-IDF features...")
    X_tfidf, vectorizer = build_tfidf(df["full_text"], fit=fit, vectorizer=vectorizer)

    # 3) SBERT
    if verbose:
        logger.info("Building SBERT embeddings...")
    X_sbert = build_sbert_embeddings(df["full_text"], model=sbert_model, show_progress=verbose)

    # 4) Structural
    if verbose:
        logger.info("Building structural features...")
    X_struct, encoders, scaler = build_structural_features(
        df, fit=fit, encoders=encoders, scaler=scaler
    )

    # 5) Assemble
    X = assemble_features(X_tfidf, X_sbert, X_struct)

    # 6) feature names
    struct_cols = [
        "text_length",
        "word_count",
        "exclamation_count",
        "question_count",
        "caps_ratio",
        "csat_norm",
        "csat_inverted",
        "first_response_minutes_norm",
        "resolution_time_minutes_norm",
        "channel_enc",
        "ticket_type_enc",
        "status_enc",
        "inferred_severity_score",
        "severity_confidence",
        "urgency_lexicon_score",
    ]
    feature_names = get_feature_names(vectorizer, X_sbert.shape[1], struct_cols)

    # 7) optionally save artifacts
    if save_artifacts_flag and fit:
        try:
            save_feature_artifacts(vectorizer, encoders, scaler, sbert_model, path=artifacts_path)
        except Exception as e:
            logger.warning(f"Failed to save feature artifacts: {e}")

    # Return dense feature matrix plus TF-IDF sparse matrix and artifacts
    return X, X_tfidf, vectorizer, encoders, scaler, feature_names
