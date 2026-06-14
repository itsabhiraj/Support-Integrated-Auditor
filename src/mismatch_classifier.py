"""
mismatch_classifier.py — Stage 3: Train & predict the mismatch classifier.

Architecture:
  - LightGBM on mixed features (TF-IDF + SBERT + structural)
  - 5-fold stratified cross-validation on pseudo-labeled data
  - CalibratedClassifierCV (isotonic) for calibrated probabilities
  - Threshold tuning via F1-max on held-out validation fold
  - Full SHAP support for explainability
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, brier_score_loss,
)
import lightgbm as lgb

from src.config import (
    LGBM_PARAMS, CV_FOLDS, RANDOM_STATE, MODELS_DIR,
    MIN_MISMATCH_PROB, TFIDF_MAX_FEATURES,
)
from src.feature_engineer import (
    build_tfidf, build_sbert_embeddings, build_structural_features,
    assemble_features, save_feature_artifacts, load_feature_artifacts,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Optimal Threshold Search
# ──────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find probability threshold that maximises F1 on validation data."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.80, 0.01):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


# ──────────────────────────────────────────────────────────────────────────────
# Feature Building Helper
# ──────────────────────────────────────────────────────────────────────────────

def build_features_for_df(
    df: pd.DataFrame,
    sbert_model=None,
    vectorizer=None,
    encoders=None,
    scaler=None,
    fit: bool = True,
    verbose: bool = True,
):
    """
    Build full feature matrix for a dataframe.
    Returns (X, vectorizer, encoders, scaler, sbert_model, feature_names).
    """
    from sentence_transformers import SentenceTransformer
    from src.config import SBERT_MODEL_NAME

    if sbert_model is None:
        logger.info(f"Loading SBERT: {SBERT_MODEL_NAME}")
        sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    X_tfidf, vectorizer = build_tfidf(df["full_text"], fit=fit, vectorizer=vectorizer)
    X_sbert = build_sbert_embeddings(df["full_text"], model=sbert_model, show_progress=verbose)
    X_struct, encoders, scaler = build_structural_features(
        df, fit=fit, encoders=encoders, scaler=scaler
    )

    X = assemble_features(X_tfidf, X_sbert, X_struct)

    struct_cols = [
        "text_length", "word_count", "exclamation_count", "question_count", "caps_ratio",
        "csat_norm", "csat_inverted", "first_response_minutes_norm",
        "resolution_time_minutes_norm", "channel_enc", "ticket_type_enc", "status_enc",
        "inferred_severity_score", "severity_confidence", "urgency_lexicon_score",
    ]
    from src.feature_engineer import get_feature_names
    feature_names = get_feature_names(vectorizer, X_sbert.shape[1], struct_cols)

    return X, vectorizer, encoders, scaler, sbert_model, feature_names


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_classifier(
    train_df: pd.DataFrame,
    sbert_model=None,
    verbose: bool = True,
) -> dict:
    """
    Train the mismatch classifier on pseudo-labeled training data.

    Returns a dict with:
      - model            : fitted LightGBM (calibrated)
      - threshold        : optimal decision threshold
      - vectorizer       : fitted TF-IDF
      - encoders         : fitted LabelEncoders
      - scaler           : fitted RobustScaler
      - sbert_model      : loaded SentenceTransformer
      - feature_names    : list of feature names
      - cv_results       : per-fold metrics
      - oof_probs        : out-of-fold probabilities (for analysis)
    """
    if verbose:
        print(f"\n[MismatchClassifier] Training on {len(train_df):,} pseudo-labeled samples")
        print(f"  Class balance: {train_df['is_mismatch'].mean()*100:.1f}% mismatch")

    y = train_df["is_mismatch"].values

    # ── Build features (fit on training set) ─────────────────────────────
    X, vectorizer, encoders, scaler, sbert_model, feature_names = build_features_for_df(
        train_df, sbert_model=sbert_model, fit=True, verbose=verbose
    )

    # ── Cross-Validation ──────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []
    oof_probs = np.zeros(len(y), dtype=np.float32)
    thresholds = []

    if verbose:
        print(f"\n  Running {CV_FOLDS}-fold stratified CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # LightGBM dataset (native API for early stopping)
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        _vp = model.predict_proba(X_val)
        val_prob = _vp[:, 1] if _vp.ndim > 1 and _vp.shape[1] > 1 else _vp.ravel()
        oof_probs[val_idx] = val_prob

        if len(np.unique(y_val)) < 2:
            auc = float('nan')
        else:
            auc = roc_auc_score(y_val, val_prob)
        t = find_optimal_threshold(y_val, val_prob)
        preds = (val_prob >= t).astype(int)
        f1  = f1_score(y_val, preds, zero_division=0)
        pre = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        bs  = brier_score_loss(y_val, val_prob)

        cv_results.append({"fold": fold+1, "auc": auc, "f1": f1, "precision": pre,
                            "recall": rec, "brier": bs, "threshold": t})
        thresholds.append(t)

        if verbose:
            print(f"    Fold {fold+1}: AUC={auc:.4f}  F1={f1:.4f}  "
                  f"Prec={pre:.4f}  Rec={rec:.4f}  Brier={bs:.4f}  T={t:.2f}")

    # ── Final OOF metrics ─────────────────────────────────────────────────
    optimal_threshold = float(np.median(thresholds))
    oof_preds = (oof_probs >= optimal_threshold).astype(int)
    if len(np.unique(y)) >= 2:
        oof_auc = roc_auc_score(y, oof_probs)
    else:
        oof_auc = float('nan')
    oof_f1  = f1_score(y, oof_preds, zero_division=0)

    if verbose:
        print(f"\n  OOF AUC:  {oof_auc:.4f}")
        print(f"  OOF F1:   {oof_f1:.4f}")
        print(f"  Decision threshold (median across folds): {optimal_threshold:.3f}")

    # ── Retrain on full data with calibration ─────────────────────────────
    if verbose:
        print("\n  Retraining on full data with isotonic calibration...")

    base_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    calibrated_model = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
    calibrated_model.fit(X, y)

    if verbose:
        print("  ✓ Calibrated model fitted")

    # ── OOF classification report ─────────────────────────────────────────
    if verbose:
        print("\n  OOF Classification Report:")
        print(classification_report(y, oof_preds, target_names=["Aligned", "Mismatch"]))

    artifact = {
        "model":         calibrated_model,
        "threshold":     optimal_threshold,
        "vectorizer":    vectorizer,
        "encoders":      encoders,
        "scaler":        scaler,
        "sbert_model":   sbert_model,
        "feature_names": feature_names,
        "cv_results":    cv_results,
        "oof_probs":     oof_probs,
        "oof_auc":       oof_auc,
        "oof_f1":        oof_f1,
    }

    return artifact


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    df: pd.DataFrame,
    artifact: dict,
    threshold: Optional[float] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run mismatch prediction on any dataframe (must have full_text + metadata).
    Adds:
      - mismatch_probability
      - mismatch_predicted
      - confidence_level  (HIGH / MEDIUM / LOW)
    """
    df = df.copy()

    X, _, _, _, _, _ = build_features_for_df(
        df,
        sbert_model=artifact["sbert_model"],
        vectorizer=artifact["vectorizer"],
        encoders=artifact["encoders"],
        scaler=artifact["scaler"],
        fit=False,
        verbose=verbose,
    )

    _proba = artifact["model"].predict_proba(X)
    probs = _proba[:, 1] if _proba.ndim > 1 and _proba.shape[1] > 1 else _proba.ravel()
    t = threshold or artifact["threshold"]
    preds = (probs >= t).astype(int)

    df["mismatch_probability"] = probs.astype(np.float32)
    df["mismatch_predicted"]   = preds
    df["confidence_level"] = pd.cut(
        probs,
        bins=[-0.001, 0.35, 0.55, 0.75, 1.001],
        labels=["LOW", "MEDIUM", "HIGH", "VERY_HIGH"],
    ).astype(str)

    if verbose:
        n_mis = preds.sum()
        print(f"\n[MismatchClassifier] Predictions: {n_mis:,} mismatches "
              f"({100*n_mis/len(df):.1f}%) of {len(df):,} tickets")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# SHAP Explanations
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    X: np.ndarray,
    artifact: dict,
    max_samples: int = 500,
) -> np.ndarray:
    """Compute SHAP values for (a sample of) the feature matrix."""
    try:
        import shap
        # Use the underlying base LightGBM estimator from calibrated model
        base_estimator = artifact["model"].calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(base_estimator)
        X_sample = X[:max_samples]
        shap_values = explainer.shap_values(X_sample)
        # For binary, shap_values is a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return shap_values
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}. Returning zeros.")
        return np.zeros((min(len(X), max_samples), X.shape[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_model(artifact: dict, path: Path = MODELS_DIR) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save model separately (heavy object)
    with open(path / "mismatch_classifier.pkl", "wb") as f:
        pickle.dump({
            "model":         artifact["model"],
            "threshold":     artifact["threshold"],
            "feature_names": artifact["feature_names"],
            "oof_auc":       artifact["oof_auc"],
            "oof_f1":        artifact["oof_f1"],
            "cv_results":    artifact["cv_results"],
        }, f)

    save_feature_artifacts(
        artifact["vectorizer"],
        artifact["encoders"],
        artifact["scaler"],
        path=path,
    )

    # Save SBERT model name reference (don't pickle full model)
    with open(path / "sbert_model_name.txt", "w") as f:
        from src.config import SBERT_MODEL_NAME
        f.write(SBERT_MODEL_NAME)

    logger.info(f"Model saved to {path}")


def load_model(path: Path = MODELS_DIR) -> dict:
    """Load a previously saved classifier artifact."""
    path = Path(path)
    with open(path / "mismatch_classifier.pkl", "rb") as f:
        artifact = pickle.load(f)

    vectorizer, encoders, scaler = load_feature_artifacts(path)
    artifact["vectorizer"] = vectorizer
    artifact["encoders"]   = encoders
    artifact["scaler"]     = scaler

    from sentence_transformers import SentenceTransformer
    sbert_name = (path / "sbert_model_name.txt").read_text().strip()
    artifact["sbert_model"] = SentenceTransformer(sbert_name)

    logger.info(f"Model loaded from {path}")
    return artifact
