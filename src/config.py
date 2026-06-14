"""
config.py — Centralized configuration for the Support Integrity Auditor.
All hyperparameters, thresholds, model names, and column mappings live here.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
MODELS_DIR   = PROJECT_ROOT / "models"
DOSSIERS_DIR = OUTPUTS_DIR / "dossiers"

PSEUDO_LABELS_PATH = OUTPUTS_DIR / "pseudo_labels.csv"
PREDICTIONS_PATH   = OUTPUTS_DIR / "predictions.csv"
EVAL_REPORT_PATH   = OUTPUTS_DIR / "evaluation_report.txt"

# ──────────────────────────────────────────────
# Column Name Aliases (handles various dataset naming conventions)
# ──────────────────────────────────────────────
COLUMN_ALIASES = {
    "ticket_id":      ["Ticket ID", "ticket_id", "id", "ID", "TicketID"],
    "subject":        ["Ticket Subject", "Subject", "subject", "title", "Title"],
    "description":    ["Ticket Description", "Description", "description", "body", "Body", "message", "Message"],
    "priority":       ["Ticket Priority", "Priority", "priority", "Priority_Level", "urgency", "Urgency"],
    "ticket_type":    ["Ticket Type", "Type", "type", "ticket_type", "Category", "category", "Issue_Category"],
    "status":         ["Ticket Status", "Status", "status"],
    "channel":        ["Ticket Channel", "Channel", "channel", "source", "Source"],
    "resolution":     ["Resolution", "resolution", "resolution_text", "Solution"],
    "customer_name":  ["Customer Name", "Name", "customer_name"],
    "product":        ["Product Purchased", "Product", "product"],
    "csat":           ["Customer Satisfaction Rating", "CSAT", "csat", "satisfaction_rating", "Rating", "Satisfaction_Score"],
    "first_response": ["First Response Time", "first_response_time", "response_time"],
    "resolution_time":["Time to Resolution", "resolution_time", "time_to_resolution", "handle_time", "Resolution_Time_Hours"],
    "date_purchased": ["Date of Purchase", "date_purchased", "purchase_date"],

}

# ──────────────────────────────────────────────
# Priority / Severity Ordinal Mapping
# ──────────────────────────────────────────────
PRIORITY_LABEL_MAP = {
    "low":      0,
    "medium":   1,
    "high":     2,
    "critical": 3,
}
PRIORITY_RANK_TO_LABEL = {v: k for k, v in PRIORITY_LABEL_MAP.items()}

SEVERITY_THRESHOLDS = {
    "low":      (0.00, 0.30),
    "medium":   (0.30, 0.55),
    "high":     (0.55, 0.78),
    "critical": (0.78, 1.01),
}

# ──────────────────────────────────────────────
# Pseudo-Labeling
# ──────────────────────────────────────────────
MISMATCH_GAP_THRESHOLD = 1        # ordinal gap to declare mismatch
MIN_PSEUDO_LABEL_CONFIDENCE = 0.55 # min confidence to include in training

# ──────────────────────────────────────────────
# Severity Inference Weights
# ──────────────────────────────────────────────
SEVERITY_WEIGHTS = {
    "lexicon":     0.35,
    "nli":         0.35,  # set to 0 if --offline
    "structural":  0.20,
    "sentiment":   0.10,
}

# ──────────────────────────────────────────────
# NLI Model
# ──────────────────────────────────────────────
NLI_MODEL_NAME    = "facebook/bart-large-mnli"
NLI_BATCH_SIZE    = 8
SBERT_MODEL_NAME  = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────
TFIDF_MAX_FEATURES = 3000
CV_FOLDS           = 5
RANDOM_STATE       = 42

LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "n_estimators":     500,
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples":20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "class_weight":     "balanced",
    "n_jobs":           -1,
    "verbose":          -1,
    "random_state":     RANDOM_STATE,
}

# ──────────────────────────────────────────────
# Evidence Dossier
# ──────────────────────────────────────────────
TOP_SHAP_FEATURES    = 5
TOP_EVIDENCE_SPANS   = 3
MIN_MISMATCH_PROB    = 0.40  # threshold to flag a ticket as mismatch

# ──────────────────────────────────────────────
# Urgency Lexicon  (curated, competition-grade)
# ──────────────────────────────────────────────
CRITICAL_SIGNALS = [
    "system down", "complete outage", "total failure", "data loss", "data breach",
    "security breach", "cannot access", "unable to access", "entirely broken",
    "production down", "server down", "database down", "payment failing",
    "all users affected", "critical error", "emergency", "urgent help",
    "immediately", "asap", "as soon as possible", "losing data", "corrupted",
    "company halted", "business stopped", "cannot function", "security vulnerability",
    "ransomware", "unauthorized access", "credentials stolen", "severe impact",
    "multiple users affected", "revenue loss", "sla breach", "deadline missed",
]

HIGH_SIGNALS = [
    "not working", "broken", "major issue", "significant problem", "disrupted",
    "keeps failing", "repeated failure", "very slow", "severely degraded",
    "several users", "team affected", "functionality lost", "error repeatedly",
    "blocked", "cannot complete", "overdue", "escalate", "need urgent",
    "frustrated", "unacceptable", "very disappointed", "angry",
]

MEDIUM_SIGNALS = [
    "problem", "issue", "error", "bug", "glitch", "not responding",
    "sometimes fails", "intermittent", "slow", "delayed", "minor disruption",
    "unexpected behavior", "wrong result", "incorrect", "workaround needed",
]

LOW_SIGNALS = [
    "question", "inquiry", "how to", "wondering", "could you please",
    "request for information", "general question", "curious", "help me understand",
    "feature request", "enhancement", "suggestion", "when will", "nice to have",
    "minor", "cosmetic", "typo", "spelling",
]

NEGATION_WORDS = [
    "not", "no", "never", "cannot", "can't", "won't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "nothing", "nowhere", "nobody",
]

NLI_SEVERITY_LABELS = [
    "This is a critical emergency requiring immediate attention",
    "This is a high priority serious issue",
    "This is a medium priority issue",
    "This is a low priority minor inquiry",
]
