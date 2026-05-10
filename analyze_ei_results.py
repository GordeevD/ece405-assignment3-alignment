#!/usr/bin/env python3
"""
Script to analyze and visualize Expert Iteration results from W&B.
Generates comparison plots and summary statistics.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

try:
    import wandb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError:
    print("Error: Required packages not installed. Install with:")
    print("  pip install wandb matplotlib numpy pandas")
    exit(1)


def fetch_ei_runs(project: str, entity: Optional[str] = None, filters: dict = None) -> list:
    """Fetch all Expert Iteration runs from a W&B project."""
    api = wandb.Api()
    
    query_filter = {"$and": [{"state": "finished"}]}
    if filters:
        query_filter["$and"].append(filters)
    
    project_path = f"{entity}/{project}" if entity else project
    runs = api.runs(project_path, filters=query_filter)
    
    return list(runs)


def extract_metrics(run) -> dict:
    """Extract key metrics from a W&B run."""
    history = run.history(samples=10000)
    
    metrics = {
        "run_name": run.name,
        "run_id": run.id,
        "config": run.config,
    }
    
    # Extract accuracy progression
    if "eval/accuracy" in history.columns:
        accs = history["eval/accuracy"].dropna()
        if len(accs) > 0:
            metrics["accuracies"] = accs.tolist()
            metrics["final_accuracy"] = float(accs.iloc[-1])
            metrics["max_accuracy"] = float(accs.max())
            metrics["accuracy_steps"] = list(range(len(accs)))
    
    # Extract entropy progression
    if "train/entropy" in history.columns:
        entropies = history["train/entropy"].dropna()
        if len(entropies) > 0:
            metrics["entropies"] = entropies.tolist()
            metrics["final_entropy"] = float(entropies.iloc[-1])
            metrics["initial_entropy"] = float(entropies.iloc[0])
    
    # Extract loss progression
    if "train/loss" in history.columns:
        losses = history["train/loss"].dropna()
        if len(losses) > 0:
            metrics["losses"] = losses.tolist()
            metrics["final_loss"] = float(losses.iloc[-1])
            metrics["initial_loss"] = float(losses.iloc[0])
    
    return metrics


def plot_accuracy_comparison(runs_metrics: list[dict], save_path: str = "accuracy_comparison.png") -> None:
    """Plot accuracy curves for multiple runs."""
    plt.figure(figsize=(12, 7))
    
    for run_metrics in runs_metrics:
        if "accuracy_steps" not in run_metrics:
            continue
        
        label = run_metrics["run_name"]
        config = run_metrics["config"]
        
        # Create informative label with hyperparameters
        label_with_config = (
            f"{label}\n"
            f"(batch={config.get('ei_batch_size', 'N/A')}, "
            f"G={config.get('g_rollouts', 'N/A')}, "
            f"epochs={config.get('sft_epochs', 'N/A')})"
        )
        
        plt.plot(
            run_metrics["accuracy_steps"],
            run_metrics["accuracies"],
            marker='o',
            label=label_with_config,
            linewidth=2,
            markersize=6
        )
    
    plt.xlabel("Expert Iteration Step", fontsize=12)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.title("Expert Iteration: Accuracy Comparison", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Accuracy comparison plot saved to {save_path}")
    plt.close()


def plot_entropy_comparison(runs_metrics: list[dict], save_path: str = "entropy_comparison.png") -> None:
    """Plot entropy curves for multiple runs."""
    fig, axes = plt.subplots(1, len(runs_metrics), figsize=(15, 5))
    
    if len(runs_metrics) == 1:
        axes = [axes]
    
    for ax, run_metrics in zip(axes, runs_metrics):
        if "entropies" not in run_metrics:
            continue
        
        ax.plot(
            range(len(run_metrics["entropies"])),
            run_metrics["entropies"],
            color='steelblue',
            linewidth=2,
            marker='o',
            markersize=3
        )
        
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Average Token Entropy", fontsize=11)
        ax.set_title(f"{run_metrics['run_name']}\nEntropy Over Training", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Entropy comparison plot saved to {save_path}")
    plt.close()


def generate_summary_table(runs_metrics: list[dict], save_path: str = "summary.csv") -> None:
    """Generate a summary table of all runs."""
    summary_data = []
    
    for run_metrics in runs_metrics:
        row = {
            "Run Name": run_metrics["run_name"],
            "Batch Size": run_metrics["config"].get("ei_batch_size", "N/A"),
            "Rollouts (G)": run_metrics["config"].get("g_rollouts", "N/A"),
            "SFT Epochs": run_metrics["config"].get("sft_epochs", "N/A"),
            "Temperature": run_metrics["config"].get("temperature", "N/A"),
            "Final Accuracy (%)": run_metrics.get("final_accuracy", "N/A"),
            "Max Accuracy (%)": run_metrics.get("max_accuracy", "N/A"),
            "Initial Entropy": run_metrics.get("initial_entropy", "N/A"),
            "Final Entropy": run_metrics.get("final_entropy", "N/A"),
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(save_path, index=False)
    print(f"✓ Summary table saved to {save_path}")
    print("\nSummary:")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Expert Iteration results from Weights & Biases"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="ece405-expert-iteration",
        help="W&B project name"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (user/team)"
    )
    parser.add_argument(
        "--run-filter",
        type=str,
        default=None,
        help="JSON filter for run names (e.g., '{\"name\": {\"$contains\": \"batch512\"}}')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ei_analysis",
        help="Directory to save analysis plots and tables"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching runs from W&B project: {args.project}")
    
    # Parse filter if provided
    filters = None
    if args.run_filter:
        try:
            filters = json.loads(args.run_filter)
        except json.JSONDecodeError:
            print(f"Error parsing filter: {args.run_filter}")
            return
    
    # Fetch runs
    runs = fetch_ei_runs(args.project, args.entity, filters)
    print(f"Found {len(runs)} runs")
    
    if len(runs) == 0:
        print("No runs found. Check your project name and filter.")
        return
    
    # Extract metrics
    print("Extracting metrics...")
    runs_metrics = []
    for run in runs:
        try:
            metrics = extract_metrics(run)
            runs_metrics.append(metrics)
            print(f"  ✓ {run.name}")
        except Exception as e:
            print(f"  ✗ {run.name}: {e}")
    
    if len(runs_metrics) == 0:
        print("No metrics extracted. Check your W&B runs.")
        return
    
    # Generate visualizations
    print("\nGenerating analysis...")
    plot_accuracy_comparison(runs_metrics, output_dir / "accuracy_comparison.png")
    plot_entropy_comparison(runs_metrics, output_dir / "entropy_comparison.png")
    generate_summary_table(runs_metrics, output_dir / "summary.csv")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
