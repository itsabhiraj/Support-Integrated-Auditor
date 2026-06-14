# 🔍 Support Integrity Auditor (SIA)

A mismatch occurs when the human-assigned ticket priority contradicts the objective severity inferred from ticket content. SIA catches both **under-prioritised critical issues** (escalation risk) and **over-prioritised trivial tickets** (resource waste).

---

## Architecture (4-Stage Pipeline)

```
Raw CRM Ticket
      │
      ▼
┌─────────────────────────────────────┐
│ Stage 1: Severity Inference Engine  │  Lexicon + NLI (zero-shot) + Structural + Sentiment
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ Stage 2: Pseudo-Label Generation    │  Compare inferred severity vs assigned priority
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ Stage 3: Mismatch Classifier (ML)   │  LightGBM + TF-IDF + SBERT + calibration
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ Stage 4: Evidence Dossier Generator │  SHAP + verbatim span extraction + Markdown reports
└─────────────────────────────────────┘
```

---

## Project Structure

```
Support Integrity Auditor/
├── data/                   ← Place your CSV here
│   └── support_tickets.csv
├── outputs/
│   ├── pseudo_labels.csv   ← Stage 2 output
│   ├── predictions.csv     ← Stage 3 output
│   ├── evaluation_report.txt
│   ├── *.png               ← Evaluation plots
│   └── dossiers/
│       ├── TKT-000001.json ← Per-ticket machine-readable dossier
│       └── TKT-000001.md   ← Per-ticket human-readable report
├── models/                 ← Saved model artifacts
├── src/
│   ├── config.py           ← All constants, hyperparams, lexicons
│   ├── data_loader.py      ← Data loading & preprocessing
│   ├── feature_engineer.py ← TF-IDF, SBERT, structural features
│   ├── severity_inferrer.py← Stage 1: Objective severity inference
│   ├── pseudo_labeler.py   ← Stage 2: Pseudo-label generation
│   ├── mismatch_classifier.py ← Stage 3: LightGBM classifier
│   ├── evidence_dossier.py ← Stage 4: Dossier generation
│   └── evaluate.py         ← Metrics & evaluation report
└── pipeline.py             ← CLI entry point
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data (if not already present)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

> **Dataset**: Download the *Customer Support Tickets* CRM dataset from Kaggle and place it as `data/support_tickets.csv`.

---

## Usage

### Run Full Pipeline (recommended)

```bash
# With NLI model (requires internet + ~1.5GB RAM for model)
python pipeline.py --data data/support_tickets.csv --stage all

# Offline mode (no NLI model; uses lexicon + structural + sentiment only)
python pipeline.py --data data/support_tickets.csv --stage all --offline
```

### Run Individual Stages

```bash
# Stage 1+2: Infer severity and generate pseudo-labels
python pipeline.py --data data/support_tickets.csv --stage infer

# Stage 3: Train classifier (after infer)
python pipeline.py --data data/support_tickets.csv --stage train

# Stage 4: Run predictions
python pipeline.py --data data/support_tickets.csv --stage predict

# Stage 5: Generate evidence dossiers
python pipeline.py --data data/support_tickets.csv --stage dossier

# Stage 6: Evaluation report
python pipeline.py --data data/support_tickets.csv --stage evaluate
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--data` | Path to CSV dataset | *(required)* |
| `--stage` | `all / infer / train / predict / dossier / evaluate` | `all` |
| `--offline` | Disable NLI model | `False` |
| `--shap` | Compute SHAP for dossiers | `True` |
| `--threshold` | Custom decision threshold [0-1] | Auto (F1-max) |
| `--model` | Path to saved model dir | `models/` |
| `--min-confidence` | Min pseudo-label confidence for training | `0.55` |
| `--quiet` | Suppress verbose output | `False` |

---

## Key Design Decisions

### Severity Inference (Stage 1)
- **Lexicon**: 60+ curated urgency signals across 4 severity levels with negation-awareness
- **NLI Zero-shot**: `facebook/bart-large-mnli` assesses severity from ticket text
- **Structural**: CSAT, resolution time, response time as proxy signals
- **Sentiment**: VADER compound score + exclamation/CAPS boosters
- Ensemble-weighted, confidence-scored per ticket

### Pseudo-Labeling Quality Gate (Stage 2)
- Only tickets with `confidence >= 0.55` are used for training
- "Uncertain" tickets are excluded, preserving label quality
- Mismatch direction tracked: `UNDER-PRIORITISED` vs `OVER-PRIORITISED`

### Classifier (Stage 3)
- **LightGBM** on mixed features: TF-IDF (3000 feats) + SBERT (384-dim) + structural (15 feats)
- 5-fold stratified CV, calibrated with isotonic regression
- F1-maximising threshold tuned per fold

### Evidence Dossiers (Stage 4)
- **Zero hallucination**: all evidence quotes are verbatim text from the ticket
- SHAP attributions with human-readable feature names
- JSON (machine-readable) + Markdown (human-readable) per ticket

---

## Expected Outputs

After a full run:

| File | Description |
|------|-------------|
| `outputs/pseudo_labels.csv` | Severity scores + pseudo-labels for all tickets |
| `outputs/predictions.csv` | Classifier predictions + probabilities |
| `outputs/evaluation_report.txt` | CV metrics, sanity checks, adversarial analysis |
| `outputs/roc_curve.png` | ROC curve |
| `outputs/confusion_matrix.png` | Confusion matrix |
| `outputs/priority_vs_severity_heatmap.png` | Assigned vs inferred priority heatmap |
| `outputs/dossiers/TKT-*.json` | Per-mismatch-ticket JSON dossier |
| `outputs/dossiers/TKT-*.md` | Per-mismatch-ticket Markdown report |

---

## Evaluation Targets

| Metric | Target |
|--------|--------|
| OOF AUC-ROC | > 0.72 |
| OOF F1 Score | > 0.60 |
| Brier Score | < 0.20 |
| Pseudo-label quality (HIGH) | > 30% of labels |
