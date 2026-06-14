"""
evidence_dossier.py — Stage 4: Generate hallucination-free evidence dossiers.

Per-ticket output:
  - Assigned vs inferred priority with gap analysis
  - Top-N SHAP feature attributions (human-readable)
  - Verbatim evidence spans from actual ticket text
  - Recommended action: ESCALATE / OK / INVESTIGATE
  - Both JSON and Markdown formats
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    TOP_SHAP_FEATURES,
    TOP_EVIDENCE_SPANS,
    MIN_MISMATCH_PROB,
    DOSSIERS_DIR,
    CRITICAL_SIGNALS, HIGH_SIGNALS, MEDIUM_SIGNALS,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Evidence Span Extractor (hallucination-free: verbatim only)
# ──────────────────────────────────────────────────────────────────────────────

_ALL_SIGNALS = [
    (3, CRITICAL_SIGNALS),
    (2, HIGH_SIGNALS),
    (1, MEDIUM_SIGNALS),
]


def extract_evidence_spans(text: str, top_n: int = TOP_EVIDENCE_SPANS) -> list[dict]:
    """
    Extract verbatim evidence spans from ticket text.
    Only returns text that physically appears in the ticket — no synthesis.

    Returns list of dicts:
      {span: str, signal_level: int, context: str}
    """
    if not text or pd.isna(text):
        return []

    text_str = str(text)
    spans_found = []

    for level, signals in _ALL_SIGNALS:
        for phrase in signals:
            pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
            for m in pattern.finditer(text_str):
                # Extract surrounding context (30 chars each side)
                start = max(0, m.start() - 30)
                end   = min(len(text_str), m.end() + 30)
                context_raw = text_str[start:end].strip()
                # Add ellipsis indicators
                prefix = "..." if start > 0 else ""
                suffix = "..." if end < len(text_str) else ""
                context = f"{prefix}{context_raw}{suffix}"

                spans_found.append({
                    "span":         m.group(),                 # verbatim match
                    "verbatim_context": context,               # surrounding text
                    "signal_level": level,
                    "char_position": m.start(),
                })

    # Deduplicate by span text, keep highest signal level
    seen = {}
    for s in spans_found:
        key = s["span"].lower()
        if key not in seen or s["signal_level"] > seen[key]["signal_level"]:
            seen[key] = s

    # Sort by signal level descending, then by position
    sorted_spans = sorted(seen.values(), key=lambda x: (-x["signal_level"], x["char_position"]))
    return sorted_spans[:top_n]


# ──────────────────────────────────────────────────────────────────────────────
# SHAP Attribution → Human-readable Feature Importances
# ──────────────────────────────────────────────────────────────────────────────

def _humanise_feature_name(name: str) -> str:
    """Convert internal feature name to user-readable form."""
    if name.startswith("tfidf:"):
        return f'Text keyword: "{name[6:]}"'
    if name.startswith("sbert_"):
        return "Semantic content embedding"
    name_map = {
        "text_length":                   "Ticket text length",
        "word_count":                    "Word count",
        "exclamation_count":             "Exclamation marks count",
        "question_count":                "Question marks count",
        "caps_ratio":                    "Proportion of uppercase letters",
        "csat_norm":                     "Customer satisfaction score",
        "csat_inverted":                 "Low customer satisfaction signal",
        "first_response_minutes_norm":   "First response time",
        "resolution_time_minutes_norm":  "Resolution time",
        "channel_enc":                   "Ticket channel (email/chat/phone)",
        "ticket_type_enc":               "Ticket type/category",
        "status_enc":                    "Ticket status",
        "inferred_severity_score":       "Inferred severity score (ensemble)",
        "severity_confidence":           "Severity inference confidence",
        "urgency_lexicon_score":         "Urgency keyword score",
    }
    return name_map.get(name, name.replace("_", " ").title())


def get_shap_attributions(
    row_shap: np.ndarray,
    feature_names: list[str],
    top_n: int = TOP_SHAP_FEATURES,
) -> list[dict]:
    """
    Return top-N SHAP attributions for a single row.
    Each is: {feature, human_name, shap_value, direction}
    """
    if row_shap is None or len(row_shap) == 0:
        return []

    indices = np.argsort(np.abs(row_shap))[::-1][:top_n]
    attributions = []
    for idx in indices:
        sv = float(row_shap[idx])
        fn = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        attributions.append({
            "feature":      fn,
            "human_name":   _humanise_feature_name(fn),
            "shap_value":   round(sv, 4),
            "direction":    "↑ Towards Mismatch" if sv > 0 else "↓ Away from Mismatch",
        })
    return attributions


# ──────────────────────────────────────────────────────────────────────────────
# Recommendation Engine
# ──────────────────────────────────────────────────────────────────────────────

def _recommend_action(
    is_mismatch: int,
    mismatch_prob: float,
    direction: str,
    inferred_severity: str,
    assigned_priority: str,
) -> tuple[str, str]:
    """Return (action, rationale) pair."""
    if not is_mismatch or mismatch_prob < MIN_MISMATCH_PROB:
        return "OK", "Assigned priority aligns with inferred ticket severity."

    if direction == "UNDER-PRIORITISED":
        if inferred_severity == "critical":
            return (
                "ESCALATE IMMEDIATELY",
                f"Ticket content signals CRITICAL severity but was assigned {assigned_priority.upper()}. "
                f"Immediate re-prioritisation required."
            )
        return (
            "ESCALATE",
            f"Ticket content indicates {inferred_severity.upper()} severity but was assigned "
            f"{assigned_priority.upper()}. Priority should be raised."
        )

    if direction == "OVER-PRIORITISED":
        return (
            "INVESTIGATE",
            f"Ticket was assigned {assigned_priority.upper()} but content signals only "
            f"{inferred_severity.upper()} severity. Review for priority inflation."
        )

    return "INVESTIGATE", "Priority gap detected. Manual review recommended."


# ──────────────────────────────────────────────────────────────────────────────
# Dossier Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_dossier(
    row: pd.Series,
    shap_values: Optional[np.ndarray],
    feature_names: Optional[list],
) -> dict:
    """
    Build a single ticket dossier dict.
    All text evidence is sourced verbatim from the ticket.
    """
    ticket_id      = str(row.get("ticket_id", "UNKNOWN"))
    assigned_prio  = str(row.get("priority", "unknown")).lower()
    inferred_sev   = str(row.get("inferred_severity", "unknown")).lower()
    severity_score = float(row.get("inferred_severity_score", 0.0))
    sev_conf       = float(row.get("severity_confidence", 0.0))
    mismatch_prob  = float(row.get("mismatch_probability", 0.0))
    is_mismatch    = int(row.get("mismatch_predicted", 0))
    direction      = str(row.get("mismatch_direction", "N/A"))
    priority_gap   = int(row.get("priority_gap", 0))
    conf_level     = str(row.get("confidence_level", "MEDIUM"))

    full_text = str(row.get("full_text", ""))

    action, rationale = _recommend_action(
        is_mismatch, mismatch_prob, direction, inferred_sev, assigned_prio
    )

    evidence_spans = extract_evidence_spans(full_text, top_n=TOP_EVIDENCE_SPANS)

    if shap_values is not None and feature_names is not None:
        shap_attrs = get_shap_attributions(shap_values, feature_names)
    else:
        shap_attrs = []

    dossier = {
        "ticket_id":            ticket_id,
        "timestamp":            "2026-06-11T22:27:22+05:30",
        "assigned_priority":    assigned_prio.capitalize(),
        "inferred_severity":    inferred_sev.capitalize(),
        "severity_score":       round(severity_score, 4),
        "severity_confidence":  round(sev_conf, 4),
        "priority_gap":         priority_gap,
        "mismatch_detected":    bool(is_mismatch),
        "mismatch_probability": round(mismatch_prob, 4),
        "mismatch_direction":   direction,
        "confidence_level":     conf_level,
        "recommended_action":   action,
        "action_rationale":     rationale,
        "evidence_spans":       evidence_spans,
        "shap_attributions":    shap_attrs,
        "ticket_subject":       str(row.get("subject", "")),
        "ticket_snippet":       full_text[:300] + ("..." if len(full_text) > 300 else ""),
    }
    return dossier


# ──────────────────────────────────────────────────────────────────────────────
# Markdown Renderer
# ──────────────────────────────────────────────────────────────────────────────

def render_markdown_dossier(d: dict) -> str:
    """
    Render a dossier dict as a human-readable Markdown report.
    No hallucination: all evidence quoted from actual ticket text.
    """
    mismatch_emoji = "🔴" if d["mismatch_detected"] else "🟢"
    action_emoji = {
        "ESCALATE IMMEDIATELY": "🚨",
        "ESCALATE": "⬆️",
        "INVESTIGATE": "🔍",
        "OK": "✅",
    }.get(d["recommended_action"], "❓")

    lines = [
        f"# Support Integrity Dossier — Ticket {d['ticket_id']}",
        f"",
        f"---",
        f"",
        f"## {mismatch_emoji} Priority Mismatch Assessment",
        f"",
        f"| Field                | Value |",
        f"|----------------------|-------|",
        f"| **Assigned Priority**   | `{d['assigned_priority']}` |",
        f"| **Inferred Severity**   | `{d['inferred_severity']}` |",
        f"| **Severity Score**      | `{d['severity_score']:.4f}` |",
        f"| **Severity Confidence** | `{d['severity_confidence']:.4f}` |",
        f"| **Priority Gap**        | `{d['priority_gap']:+d}` |",
        f"| **Mismatch Detected**   | `{'YES' if d['mismatch_detected'] else 'NO'}` |",
        f"| **Mismatch Probability**| `{d['mismatch_probability']:.4f}` |",
        f"| **Direction**           | `{d['mismatch_direction']}` |",
        f"| **Confidence Level**    | `{d['confidence_level']}` |",
        f"",
        f"---",
        f"",
        f"## {action_emoji} Recommended Action: **{d['recommended_action']}**",
        f"",
        f"> {d['action_rationale']}",
        f"",
        f"---",
        f"",
        f"## 📋 Ticket Summary",
        f"",
        f"**Subject:** {d['ticket_subject']}",
        f"",
        f"**Excerpt:**",
        f"> {d['ticket_snippet']}",
        f"",
        f"---",
        f"",
        f"## 🔍 Evidence Spans (Verbatim from Ticket)",
        f"",
        f"_All quotes below are copied verbatim from the ticket text — no text has been generated or summarised._",
        f"",
    ]

    if d["evidence_spans"]:
        for i, span in enumerate(d["evidence_spans"], 1):
            level_label = {3: "CRITICAL", 2: "HIGH", 1: "MEDIUM"}.get(span["signal_level"], "")
            lines.append(f"**{i}. [{level_label} signal]** `\"{span['span']}\"`")
            lines.append(f"")
            lines.append(f"   Context: *\"{span['verbatim_context']}\"*")
            lines.append(f"")
    else:
        lines.append("_No direct urgency signals detected in ticket text._")
        lines.append("")

    lines += [
        f"---",
        f"",
        f"## 📊 Top Feature Attributions (SHAP)",
        f"",
        f"_These are the model features with the highest influence on the mismatch prediction._",
        f"",
    ]

    if d["shap_attributions"]:
        lines.append("| Rank | Feature | SHAP Value | Direction |")
        lines.append("|------|---------|------------|-----------|")
        for i, attr in enumerate(d["shap_attributions"], 1):
            lines.append(
                f"| {i} | {attr['human_name']} | `{attr['shap_value']:+.4f}` | {attr['direction']} |"
            )
    else:
        lines.append("_SHAP attributions not available (requires classifier run with --shap flag)._")

    lines += [
        "",
        "---",
        "",
        f"*Generated by Support Integrity Auditor (SIA) — {d['timestamp']}*",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Batch Dossier Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_dossiers(
    df: pd.DataFrame,
    shap_values: Optional[np.ndarray],
    feature_names: Optional[list],
    output_dir: Path = DOSSIERS_DIR,
    only_mismatches: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Generate and save dossiers for all (or mismatch-only) tickets.

    Returns list of dossier dicts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if only_mismatches and "mismatch_predicted" in df.columns:
        target_df = df[df["mismatch_predicted"] == 1].copy()
    else:
        target_df = df.copy()

    target_df = target_df.reset_index(drop=True)
    N = len(target_df)

    if verbose:
        print(f"\n[EvidenceDossier] Generating dossiers for {N:,} tickets...")

    dossiers = []
    for i, (_, row) in enumerate(target_df.iterrows()):
        row_shap = shap_values[i] if (shap_values is not None and i < len(shap_values)) else None
        d = build_dossier(row, row_shap, feature_names)
        dossiers.append(d)

        # Save JSON
        json_path = output_dir / f"{d['ticket_id']}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, default=str)

        # Save Markdown
        md_path = output_dir / f"{d['ticket_id']}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(render_markdown_dossier(d))

    if verbose:
        mismatches = sum(1 for d in dossiers if d["mismatch_detected"])
        print(f"  ✓ {N:,} dossiers saved to {output_dir}")
        print(f"  Mismatches in dossiers: {mismatches:,}")

    return dossiers
