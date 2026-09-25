import os
import glob
from tbparse import SummaryReader
import pandas as pd

def get_run_metrics(file_path):
    try:
        reader = SummaryReader(file_path, extra_columns={'wall_time'})
        df = reader.scalars
        
        if df.empty:
            return None
        
        # --- DEBUG: Print all unique tags found in this event file ---
        seed_name = os.path.basename(os.path.dirname(file_path))
        print(f"\n  [{seed_name}] Available TensorBoard Tags:")
        print(df["tag"].unique().tolist())
        print("-" * 50)
        
        metrics = {}
        
        # 1. Wall-clock duration
        if "wall_time" in df.columns:
            start_time = df["wall_time"].min()
            end_time = df["wall_time"].max()
            metrics["duration"] = end_time - start_time
        else:
            metrics["duration"] = None
            
        # 2. Extract logged resource metrics if available
        resource_tags = {
            "cpu_mem": "Resource / CPU Memory (MB)",
            "gpu_alloc": "Resource / GPU Memory Allocated (MB)",
            "gpu_res": "Resource / GPU Memory Reserved (MB)",
            "sps": "Resource / Steps Per Second (SPS)"
        }
        
        for key, tag in resource_tags.items():
            tag_df = df[df["tag"] == tag]
            if not tag_df.empty:
                if "Memory" in tag:
                    metrics[key] = tag_df["value"].max()
                else:
                    metrics[key] = tag_df["value"].mean()
            else:
                metrics[key] = None
                
        return metrics
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return None

def main():
    base_dir = "h1_flat"
    all_event_files = glob.glob(os.path.join(base_dir, "**/events.out*"), recursive=True)
    dac_files = sorted([f for f in all_event_files if "dac" in f.lower()])

    print("=" * 80)
    print("TAG DIAGNOSTIC REPORT")
    print("=" * 80)

    for path in dac_files:
        get_run_metrics(path)

    print("=" * 80)

if __name__ == "__main__":
    main()
