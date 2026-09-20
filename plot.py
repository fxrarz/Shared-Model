import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader

# -----------------------------------------------------------------------------
# 1. Helper function to scan log files and list all unique tags
# -----------------------------------------------------------------------------
def print_available_tags(file_paths):
    for path in file_paths:
        if os.path.exists(path):
            try:
                reader = SummaryReader(path)
                df = reader.scalars
                if not df.empty:
                    tags = df["tag"].unique()
                    print(f"\n--- Available Tags in {path} ---")
                    for t in sorted(tags):
                        print(f"  - {t}")
                    return # Found one valid file, listing its tags is enough
            except Exception as e:
                continue
    print("No valid tfevents files found to extract tags.")

# -----------------------------------------------------------------------------
# 2. Helper function to load and process tfevents files for any given metric tag
# -----------------------------------------------------------------------------
def load_seed_metric(file_paths, target_tag):
    seed_dfs = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"Warning: File not found -> {path}")
            continue
        reader = SummaryReader(path)
        df = reader.scalars
        # Filter for the requested metric tag
        metric_df = df[df["tag"] == target_tag]
        if not metric_df.empty:
            seed_dfs.append(metric_df[["step", "value"]].copy())
    return seed_dfs

# -----------------------------------------------------------------------------
# 3. Helper function to aggregate seeds and plot mean + optional std
# -----------------------------------------------------------------------------
def plot_aggregated_performance(seed_dfs_list, label, color, show_ci=True):
    if not seed_dfs_list:
        print(f"No data available for {label}")
        return

    # Find global min/max steps across all seeds to create a unified common step grid
    min_step = min(df["step"].min() for df in seed_dfs_list)
    max_step = max(df["step"].max() for df in seed_dfs_list)
    common_steps = np.linspace(min_step, max_step, 200)

    interpolated_values = []
    for df in seed_dfs_list:
        # Interpolate each seed's values onto the common step grid
        interp_vals = np.interp(common_steps, df["step"], df["value"])
        interpolated_values.append(interp_vals)

    interpolated_values = np.array(interpolated_values)
    
    # Calculate mean and standard deviation across seeds
    mean = np.mean(interpolated_values, axis=0)
    std = np.std(interpolated_values, axis=0)

    # Plot mean line
    plt.plot(common_steps, mean, label=label, color=color, linewidth=1)
    
    # Conditionally plot shaded confidence interval band
    if show_ci:
        plt.fill_between(common_steps, mean - std, mean + std, color=color, alpha=0.2)

# -----------------------------------------------------------------------------
# 4. Main execution with Argument Parsing
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot multi-seed TensorBoard metrics for PPO architectures.")
    parser.add_argument(
        "--tag", 
        type=str, 
        default="Reward / Total reward (mean)", 
        help="The exact TensorBoard tag to plot"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Filename for the saved plot image"
    )
    parser.add_argument(
        "--no-ci", 
        action="store_true", 
        help="Disable the confidence interval (shaded standard deviation) band"
    )
    parser.add_argument(
        "--list-tags", 
        action="store_true", 
        help="Print all available TensorBoard tags from your log files and exit"
    )
    args = parser.parse_args()

    # Discover log files
    base_dir = "h1_flat"
    dac_files = glob.glob(f"{base_dir}/DAC*/events.out*")
    uac_files = glob.glob(f"{base_dir}/UAC*/events.out*")
    all_files = dac_files + uac_files

    # If user wants to inspect available tags, list them and exit
    if args.list_tags:
        print_available_tags(all_files)
        return

    print(f"Target Metric Tag: {args.tag}")
    print(f"Confidence Interval Enabled: {not args.no_ci}")
    
    print("Loading Architecture DAC logs...")
    dac_dfs = load_seed_metric(dac_files, args.tag)

    print("Loading Architecture UAC logs...")
    uac_dfs = load_seed_metric(uac_files, args.tag)

    # Set up publication-style plot layout
    plt.figure(figsize=(5, 5), dpi=300)

    # Plot both architectures with CI toggle option
    plot_aggregated_performance(dac_dfs, label="DAC", color="#1f77b4", show_ci=not args.no_ci)
    plot_aggregated_performance(uac_dfs, label="UAC", color="#ff7f0e", show_ci=not args.no_ci)

    # Styling the plot for a paper
    plt.title(f"Comparison: {args.tag} (5 Seeds)", fontsize=11, fontweight='bold')
    plt.xlabel("Total Environment Steps", fontsize=10)
    plt.ylabel(args.tag, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="best")
    plt.tight_layout()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        safe_tag_name = args.tag.lower().replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        output_path = f"{safe_tag_name}_comparison.png"

    # Save figure for your paper
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
