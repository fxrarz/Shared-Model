# python3 plot_loss.py --tag "Loss / Policy loss" --ylabel "Policy Loss"
# python3 plot_loss.py --tag "Loss / Policy loss" --ylabel "Policy Loss" --layout vertical
# python3 plot_loss.py --tag "Loss / Policy loss" --ylabel "Policy Loss" --layout separate

import os
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tbparse import SummaryReader

# Suppress TensorFlow C++ logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -----------------------------------------------------------------------------
# 1. Define Run IDs for each configuration
# -----------------------------------------------------------------------------
#FLAT_DAC_IDS = ["hngeh5y3", "pg19gtz2", "jwhenxkf", "axa3kayn", "1p1awscx"]
FLAT_DAC_IDS = ["yj4mwtud", "4ke6dhsp", "9k3l9rzu", "9k3l9rzu", "02i6vy7m"]
FLAT_UAC_IDS = ["bg8rgja1", "c32see1z", "suxmen6e", "qt3golzl", "1avfd3ol"]

#STAIR_DAC_IDS = ["69nnus8b", "onpwaq4w", "a332d2cb", "h7xqq02d", "bcwovr50"]
STAIR_DAC_IDS = ["8fbwql3l", "hnzg430e", "y4fpmjus", "y4fpmjus", "g1a0v6xr"]
STAIR_UAC_IDS = ["kntc8q7s", "gwvl3cr7", "y2d19akk", "nsvg62yy", "fi0ekhmz"]

# -----------------------------------------------------------------------------
# Helper: Scan and return all unique tags across all run folders
# -----------------------------------------------------------------------------
def get_available_tags(wandb_base_dir):
    unique_tags = set()
    run_dirs = glob.glob(os.path.join(wandb_base_dir, "run-*"))
    for rdir in run_dirs:
        try:
            reader = SummaryReader(rdir)
            df = reader.scalars
            if not df.empty and "tag" in df.columns:
                unique_tags.update(df["tag"].unique())
        except Exception:
            continue
    return sorted(list(unique_tags))

# -----------------------------------------------------------------------------
# 2. Helper function to extract metrics directory by directory
# -----------------------------------------------------------------------------
def load_group_metrics(wandb_base_dir, run_ids, target_tag):
    seed_dfs = []
    
    for run_id in run_ids:
        matching_dirs = glob.glob(os.path.join(wandb_base_dir, f"run-*-{run_id}"))
        
        if not matching_dirs:
            print(f"Warning: Folder not found for run ID: {run_id}")
            continue

        run_dir = matching_dirs[0]
        try:
            reader = SummaryReader(run_dir)
            df = reader.scalars
            
            if df.empty:
                print(f"Warning: No scalar data in run ID: {run_id}")
                continue
                
            run_subset = df[df["tag"] == target_tag]
            if not run_subset.empty:
                seed_dfs.append(run_subset[["step", "value"]].copy())
            else:
                available = df["tag"].unique()
                print(f"Warning: Tag '{target_tag}' not found in run ID: {run_id}. Available tags: {list(available)}")
        except Exception as e:
            print(f"Error reading logs for run ID {run_id}: {e}")

    return seed_dfs

# -----------------------------------------------------------------------------
# 3. Helper function to aggregate seeds and plot mean + 95% CI
# -----------------------------------------------------------------------------
def plot_aggregated_performance(ax, seed_dfs_list, label, color, show_ci=True):
    if not seed_dfs_list:
        print(f"No data available to plot for {label}")
        return

    min_step = min(df["step"].min() for df in seed_dfs_list)
    max_step = max(df["step"].max() for df in seed_dfs_list)
    common_steps = np.linspace(min_step, max_step, 200)

    interpolated_values = []
    for df in seed_dfs_list:
        clean_df = df.drop_duplicates(subset=["step"]).sort_values("step")
        interp_vals = np.interp(common_steps, clean_df["step"], clean_df["value"])
        interpolated_values.append(interp_vals)

    interpolated_values = np.array(interpolated_values)
    n = len(seed_dfs_list)
    
    mean = np.mean(interpolated_values, axis=0)
    
    # Calculate sample standard deviation (ddof=1) and Standard Error of the Mean (SEM)
    std = np.std(interpolated_values, axis=0, ddof=1)
    sem = std / np.sqrt(n)
    
    # Compute 95% confidence interval margin using t-distribution (df = n - 1)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_margin = t_crit * sem

    ax.plot(common_steps, mean, label=label, color=color, linewidth=1.5)
    
    if show_ci:
        ax.fill_between(common_steps, mean - ci_margin, mean + ci_margin, color=color, alpha=0.2)

# -----------------------------------------------------------------------------
# 4. Main execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot multi-seed WandB metrics comparing UAC and DAC with 95% CI.")
    parser.add_argument("--tag", type=str, default="Loss / Policy loss", help="Exact TensorBoard tag name")
    parser.add_argument("--ylabel", type=str, default=None, help="Custom y-axis label (defaults to --tag if not specified)")
    parser.add_argument("--output", type=str, default=None, help="Saved filename")
    parser.add_argument("--no-ci", action="store_true", help="Disable confidence interval band")
    parser.add_argument("--list-tags", action="store_true", help="Print available tags and exit")
    parser.add_argument("--layout", type=str, choices=["horizontal", "vertical", "separate"], default="horizontal",
                        help="Plot layout style: horizontal (side-by-side), vertical (stacked), or separate (individual files)")
    args = parser.parse_args()

    wandb_path = "wandb"
    if not os.path.exists(wandb_path):
        print(f"Error: '{wandb_path}' directory not found in current path.")
        return

    if args.list_tags:
        print("\nScanning event logs for available tags...")
        all_tags = get_available_tags(wandb_path)
        if all_tags:
            print("\n--- Available Tags in Logs ---")
            for t in all_tags:
                print(f"  - \"{t}\"")
        else:
            print("No tags found inside wandb event files.")
        return

    print(f"Target Metric Tag: \"{args.tag}\"\n")

    # Determine y-axis label name
    y_label_text = args.ylabel if args.ylabel else args.tag

    flat_dac_dfs = load_group_metrics(wandb_path, FLAT_DAC_IDS, args.tag)
    flat_uac_dfs = load_group_metrics(wandb_path, FLAT_UAC_IDS, args.tag)
    stair_dac_dfs = load_group_metrics(wandb_path, STAIR_DAC_IDS, args.tag)
    stair_uac_dfs = load_group_metrics(wandb_path, STAIR_UAC_IDS, args.tag)

    base_output = args.output or "loss_flat_vs_stair_comparison.png"

    if args.layout == "separate":
        # --- Separate Individual Saves ---
        # 1. Flat Terrain
        fig, ax_flat = plt.subplots(figsize=(6, 4.5), dpi=300)
        plot_aggregated_performance(ax_flat, flat_dac_dfs, label="DAC", color="#1f77b4", show_ci=not args.no_ci)
        plot_aggregated_performance(ax_flat, flat_uac_dfs, label="UAC", color="#ff7f0e", show_ci=not args.no_ci)
        ax_flat.set_title("Flat Terrain", fontsize=11, fontweight='bold')
        ax_flat.set_xlabel("Total Environment Steps", fontsize=10)
        ax_flat.set_ylabel(y_label_text, fontsize=10)
        ax_flat.grid(True, linestyle="--", alpha=0.5)
        if flat_dac_dfs or flat_uac_dfs:
            ax_flat.legend(fontsize=9, loc="best")
        plt.tight_layout()
        
        flat_output = f"flat_{base_output}"
        plt.savefig(flat_output, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Flat terrain plot saved to {flat_output}")

        # 2. Stairway Terrain
        fig, ax_stair = plt.subplots(figsize=(6, 4.5), dpi=300)
        plot_aggregated_performance(ax_stair, stair_dac_dfs, label="DAC", color="#1f77b4", show_ci=not args.no_ci)
        plot_aggregated_performance(ax_stair, stair_uac_dfs, label="UAC", color="#ff7f0e", show_ci=not args.no_ci)
        ax_stair.set_title("Stairway Terrain", fontsize=11, fontweight='bold')
        ax_stair.set_xlabel("Total Environment Steps", fontsize=10)
        ax_stair.set_ylabel(y_label_text, fontsize=10)
        ax_stair.grid(True, linestyle="--", alpha=0.5)
        if stair_dac_dfs or stair_uac_dfs:
            ax_stair.legend(fontsize=9, loc="best")
        plt.tight_layout()

        stair_output = f"stair_{base_output}"
        plt.savefig(stair_output, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Stairway terrain plot saved to {stair_output}")

    else:
        # --- Combined Subplots (Horizontal or Vertical) ---
        if args.layout == "vertical":
            fig, axes = plt.subplots(2, 1, figsize=(4, 7), dpi=300)
        else:  # horizontal
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

        # --- Subplot 1: Flat Terrain ---
        ax_flat = axes[0]
        plot_aggregated_performance(ax_flat, flat_dac_dfs, label="DAC", color="#1f77b4", show_ci=not args.no_ci)
        plot_aggregated_performance(ax_flat, flat_uac_dfs, label="UAC", color="#ff7f0e", show_ci=not args.no_ci)
        ax_flat.set_title("Flat Terrain", fontsize=11, fontweight='bold')
        ax_flat.set_xlabel("Total Environment Steps", fontsize=10)
        ax_flat.set_ylabel(y_label_text, fontsize=10)
        ax_flat.grid(True, linestyle="--", alpha=0.5)
        if flat_dac_dfs or flat_uac_dfs:
            leg1 = ax_flat.legend(fontsize=9, loc="best")
            for line in leg1.get_lines():
                line.set_linewidth(2.5)

        # --- Subplot 2: Stairway Terrain ---
        ax_stair = axes[1]
        plot_aggregated_performance(ax_stair, stair_dac_dfs, label="DAC", color="#1f77b4", show_ci=not args.no_ci)
        plot_aggregated_performance(ax_stair, stair_uac_dfs, label="UAC", color="#ff7f0e", show_ci=not args.no_ci)
        ax_stair.set_title("Stairway Terrain", fontsize=11, fontweight='bold')
        ax_stair.set_xlabel("Total Environment Steps", fontsize=10)
        ax_stair.set_ylabel(y_label_text, fontsize=10)
        ax_stair.grid(True, linestyle="--", alpha=0.5)
        if stair_dac_dfs or stair_uac_dfs:
            leg2 = ax_stair.legend(fontsize=9, loc="best")
            for line in leg2.get_lines():
                line.set_linewidth(2.5)

        plt.tight_layout()
        plt.savefig(base_output, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"\nPlot successfully saved to {base_output}")

if __name__ == "__main__":
    main()
