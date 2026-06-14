"""
pipeline.py — End-to-end CLI orchestrator for Support Integrity Auditor (SIA).

Usage:
  python pipeline.py --data data/support_tickets.csv --stage all
  python pipeline.py --data data/support_tickets.csv --stage all --offline
  python pipeline.py --data data/support_tickets.csv --stage infer
  python pipeline.py --data data/support_tickets.csv --stage train
  python pipeline.py --data data/support_tickets.csv --stage predict
  python pipeline.py --data data/support_tickets.csv --stage dossier
  python pipeline.py --data data/support_tickets.csv --stage evaluate
"""

import sys
import logging
import time
import pandas as pd
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    OUTPUTS_DIR, MODELS_DIR, DOSSIERS_DIR,
    PSEUDO_LABELS_PATH, PREDICTIONS_PATH
)
from src.data_loader import load_data
from src.severity_inferrer import infer_severity
from src.pseudo_labeler import generate_pseudo_labels, save_pseudo_labels, get_training_set
from src.mismatch_classifier import train_classifier, predict, save_model, load_model, compute_shap_values
from src.evidence_dossier import generate_all_dossiers
from src.evaluate import generate_evaluation_report
from src.feature_engineer import build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SIA")
console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold red]Support Integrity Auditor[/bold red] [dim](SIA)[/dim]\n"
        "[dim]Detecting Priority Mismatch in CRM Support Tickets[/dim]",
        border_style="red",
    ))


def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


# ──────────────────────────────────────────────────────────────────────────────
# Stage Runners
# ──────────────────────────────────────────────────────────────────────────────

def run_infer(data_path: str, offline: bool, verbose: bool) -> "pd.DataFrame":
    """Stage 1+2: Load data → Infer severity → Generate pseudo-labels."""
    import pandas as pd

    t0 = time.time()
    console.print("\n[bold cyan]── Stage 1: Loading Data[/bold cyan]")
    df = load_data(data_path, verbose=verbose)

    console.print("\n[bold cyan]── Stage 2: Severity Inference[/bold cyan]")
    if offline:
        console.print("  [yellow]Offline mode: NLI model disabled, using lexicon+structural+sentiment.[/yellow]")
    df = infer_severity(df, offline=offline, verbose=verbose)

    console.print("\n[bold cyan]── Stage 3: Pseudo-Label Generation[/bold cyan]")
    df = generate_pseudo_labels(df, verbose=verbose)
    save_pseudo_labels(df)
    console.print(f"  [green]✓ Pseudo-labels saved to {PSEUDO_LABELS_PATH}[/green]")
    console.print(f"  [dim]Elapsed: {elapsed(t0)}[/dim]")
    return df


def run_train(df: "pd.DataFrame", verbose: bool) -> dict:
    """Stage 3: Train mismatch classifier on pseudo-labels."""
    import pandas as pd

    t0 = time.time()
    console.print("\n[bold cyan]── Stage 4: Training Mismatch Classifier[/bold cyan]")

    train_df = get_training_set(df)
    if len(train_df) < 50:
        console.print(
            "[bold red]ERROR[/bold red]: Fewer than 50 high-confidence training samples. "
            "Consider lowering --min-confidence or checking the dataset."
        )
        sys.exit(1)

    artifact = train_classifier(train_df, verbose=verbose)
    save_model(artifact, path=MODELS_DIR)
    console.print(f"\n  [green]✓ Model saved to {MODELS_DIR}[/green]")

    # Print CV summary table
    table = Table(title="Cross-Validation Results", show_header=True, header_style="bold magenta")
    table.add_column("Fold", justify="center")
    table.add_column("AUC", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    for r in artifact["cv_results"]:
        table.add_row(str(r["fold"]), f"{r['auc']:.4f}", f"{r['f1']:.4f}",
                      f"{r['precision']:.4f}", f"{r['recall']:.4f}")
    console.print(table)
    console.print(f"\n  OOF AUC: [bold green]{artifact['oof_auc']:.4f}[/bold green]  "
                  f"OOF F1: [bold green]{artifact['oof_f1']:.4f}[/bold green]")
    console.print(f"  [dim]Elapsed: {elapsed(t0)}[/dim]")
    return artifact


def run_predict(df: "pd.DataFrame", artifact: dict, threshold: float, verbose: bool) -> "pd.DataFrame":
    """Stage 4: Run predictions on full dataset."""
    import pandas as pd

    t0 = time.time()
    console.print("\n[bold cyan]── Stage 5: Running Predictions[/bold cyan]")
    df = predict(df, artifact, threshold=threshold, verbose=verbose)

    # Save predictions
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_cols = [
        "ticket_id", "priority", "inferred_severity", "inferred_severity_score",
        "severity_confidence", "priority_gap", "is_mismatch",
        "mismatch_probability", "mismatch_predicted", "mismatch_direction",
        "confidence_level", "subject", "description",
    ]
    pred_cols = [c for c in pred_cols if c in df.columns]
    df[pred_cols].to_csv(PREDICTIONS_PATH, index=False)
    console.print(f"  [green]✓ Predictions saved to {PREDICTIONS_PATH}[/green]")
    console.print(f"  [dim]Elapsed: {elapsed(t0)}[/dim]")
    return df


def run_dossier(df: "pd.DataFrame", artifact: dict, shap: bool, verbose: bool) -> None:
    """Stage 5: Generate evidence dossiers."""
    t0 = time.time()
    console.print("\n[bold cyan]── Stage 6: Evidence Dossier Generation[/bold cyan]")

    shap_values = None
    feature_names = artifact.get("feature_names")

    if shap and "mismatch_predicted" in df.columns:
        mismatch_df = df[df["mismatch_predicted"] == 1].copy().reset_index(drop=True)
        if len(mismatch_df) > 0:
            console.print(f"  Computing SHAP for {len(mismatch_df):,} mismatch tickets...")
            try:
                X_mis, _, _, _, _, _ = build_features(
                    mismatch_df,
                    sbert_model=artifact["sbert_model"],
                    vectorizer=artifact["vectorizer"],
                    encoders=artifact["encoders"],
                    scaler=artifact["scaler"],
                    fit=False,
                    verbose=False,
                )
                shap_values = compute_shap_values(X_mis, artifact)
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")
        generate_all_dossiers(mismatch_df, shap_values, feature_names, verbose=verbose)
    else:
        generate_all_dossiers(df, None, feature_names, only_mismatches=True, verbose=verbose)

    console.print(f"  [green]✓ Dossiers saved to {DOSSIERS_DIR}[/green]")
    console.print(f"  [dim]Elapsed: {elapsed(t0)}[/dim]")


def run_evaluate(df: "pd.DataFrame", artifact: dict) -> None:
    """Generate evaluation report and plots."""
    console.print("\n[bold cyan]── Stage 7: Evaluation Report[/bold cyan]")
    generate_evaluation_report(
        df,
        cv_results=artifact.get("cv_results"),
        oof_probs=artifact.get("oof_probs"),
        threshold=artifact.get("threshold", 0.50),
        output_dir=OUTPUTS_DIR,
        verbose=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--data",          default="data/support_tickets.csv", help="Path to CRM CSV dataset")
@click.option("--stage",         default="all",
              type=click.Choice(["all", "infer", "train", "predict", "dossier", "evaluate"]),
              help="Pipeline stage to run")
@click.option("--offline",       is_flag=True, default=False, help="Disable NLI model (offline mode)")
@click.option("--shap",          is_flag=True, default=True,  help="Compute SHAP values for dossiers")
@click.option("--threshold",     default=None,  type=float,   help="Custom decision threshold [0-1]")
@click.option("--model",         default=None,               help="Path to saved model dir (for predict/dossier stages)")
@click.option("--min-confidence",default=None,  type=float,  help="Min pseudo-label confidence for training")
@click.option("--verbose/--quiet", default=True,             help="Verbosity")
def main(data, stage, offline, shap, threshold, model, min_confidence, verbose):
    """Support Integrity Auditor — Priority Mismatch Detection Pipeline."""
    print_banner()

    total_start = time.time()

    # Apply config overrides
    if min_confidence is not None:
        import src.config as cfg
        cfg.MIN_PSEUDO_LABEL_CONFIDENCE = min_confidence

    # ── Stage: infer ──────────────────────────────────────────────────────
    if stage in ("all", "infer", "train", "predict", "dossier", "evaluate"):
        df = run_infer(data, offline, verbose)

    # ── Stage: train ──────────────────────────────────────────────────────
    artifact = None
    if stage in ("all", "train"):
        artifact = run_train(df, verbose)

    # ── Load model if needed and not just trained ─────────────────────────
    if stage in ("predict", "dossier", "evaluate") and artifact is None:
        model_path = Path(model) if model else MODELS_DIR
        console.print(f"\n  Loading model from {model_path}...")
        try:
            artifact = load_model(model_path)
        except FileNotFoundError:
            console.print(
                "[bold red]ERROR[/bold red]: No saved model found. "
                "Run with --stage all or --stage train first."
            )
            sys.exit(1)

    # ── Stage: predict ────────────────────────────────────────────────────
    if stage in ("all", "predict", "dossier", "evaluate"):
        if artifact:
            df = run_predict(df, artifact, threshold, verbose)

    # ── Stage: dossier ────────────────────────────────────────────────────
    if stage in ("all", "dossier"):
        if artifact:
            run_dossier(df, artifact, shap, verbose)

    # ── Stage: evaluate ───────────────────────────────────────────────────
    if stage in ("all", "evaluate"):
        if artifact:
            run_evaluate(df, artifact)

    console.print(
        f"\n[bold green]✓ Pipeline complete in {elapsed(total_start)}[/bold green]"
    )
    console.print(f"  Outputs: [dim]{OUTPUTS_DIR}[/dim]")
    console.print(f"  Dossiers: [dim]{DOSSIERS_DIR}[/dim]")


if __name__ == "__main__":
    main()
