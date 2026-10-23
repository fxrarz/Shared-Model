import scipy.stats as stats
import numpy as np
import pandas as pd

def training_cost(col=None):
    # null: UAC is worse than or equal to DAC
    # alternative: UAC is better (faster) than DAC
    # Test: 1-tailed, unpaired Welch's t-test
    df = pd.read_csv("dataset/training_cost.csv", index_col=0)
    print(df.head(10), f"\n{'-'*25}")
    if col == None:
        return
    print("Test Column: ", col)
    print("Null Hypothesis: UAC is worse than or equal to DAC")
    print("Alternate Hypothesis: UAC is better (faster) than DAC")
    print(f"{'-'*25}")
    time_dac_flat = df[col][(df['Arch.'] == 'DAC') & (df['Terrain'] == 'Flat')].to_numpy()
    time_uac_flat = df[col][(df['Arch.'] == 'UAC') & (df['Terrain'] == 'Flat')].to_numpy()
    t_stat, p_val = stats.ttest_ind(time_uac_flat, time_dac_flat, equal_var=False, alternative='less')
    print(f"t-value: {t_stat:.5f}")
    print(f"P-value: {p_val:.5f}")
    if p_val < 0.05:
      print("Reject the Null! UAC uses significantly fewer resources than DAC on Flat terrain.")
    else:
      print("Fail to reject the Null. We cannot prove UAC uses fewer resources on Flat terrain.")
    print(f"{'-'*25}")
    time_dac_stair = df[col][(df['Arch.'] == 'DAC') & (df['Terrain'] == 'Stair')].to_numpy()
    time_uac_stair = df[col][(df['Arch.'] == 'UAC') & (df['Terrain'] == 'Stair')].to_numpy()
    t_stat, p_val = stats.ttest_ind(time_uac_stair, time_dac_stair, equal_var=False, alternative='less')
    print(f"t-value: {t_stat:.5f}")
    print(f"P-value: {p_val:.5f}")
    if p_val < 0.05:
      print("Reject the Null! UAC uses significantly fewer resources than DAC on Stair terrain.")
    else:
      print("Fail to reject the Null. We cannot prove UAC uses fewer resources on Stair terrain.")
    print(f"{'-'*25}")

training_cost('Training time (min)')
training_cost('Peak GPU VRAM (%)')
training_cost('Peak process RAM (GiB)')





def eval_reward(col=None):
    # null: UAC is worse than or equal to DAC in reward
    # alternative: UAC is better (faster) than DAC in reward
    # Test: 1-tailed, unpaired Welch's t-test
    df = pd.read_csv("dataset/eval_reward.csv", index_col=0)
    print(df.head(10), f"\n{'-'*25}")
    if col == None:
        return
    print("Test Column: ", col)
    print("Null Hypothesis: UAC is worse than or equal to DAC in reward")
    print("Alternate Hypothesis: UAC is better (faster) than DAC in reward")
    print(f"{'-'*25}")
    reward_dac_flat = df[col][(df['Arch.'] == 'DAC') & (df['Terrain'] == 'Flat')].to_numpy()
    reward_uac_flat = df[col][(df['Arch.'] == 'UAC') & (df['Terrain'] == 'Flat')].to_numpy()
    t_stat, p_val = stats.ttest_ind(reward_uac_flat, reward_dac_flat, equal_var=False, alternative='greater')
    print(f"t-value: {t_stat:.5f}")
    print(f"P-value: {p_val:.5f}")
    if p_val < 0.05:
      print("Reject the Null! UAC obtains significantly higher reward than DAC on Flat terrain.")
    else:
      print("Fail to reject the Null. We cannot prove UAC obtains significantly higher reward than DAC on Flat terrain.")
    print(f"{'-'*25}")
    reward_dac_stair = df[col][(df['Arch.'] == 'DAC') & (df['Terrain'] == 'Stair')].to_numpy()
    reward_uac_stair = df[col][(df['Arch.'] == 'UAC') & (df['Terrain'] == 'Stair')].to_numpy()
    t_stat, p_val = stats.ttest_ind(reward_uac_stair, reward_dac_stair, equal_var=False, alternative='greater')
    print(f"t-value: {t_stat:.5f}")
    print(f"P-value: {p_val:.5f}")
    if p_val < 0.05:
      print("Reject the Null! UAC obtains significantly higher reward than DAC on Stair terrain.")
    else:
      print("Fail to reject the Null. We cannot prove UAC obtains significantly higher reward than DAC on Stair terrain.")
    print(f"{'-'*25}")

eval_reward('Reward')





def push(col=None):
    # null: UAC is worse than or equal to DAC in fall
    # alternative: UAC is better (faster) than DAC in fall
    # Test: 1-tailed, unpaired Welch's t-test
    df = pd.read_csv("dataset/push.csv", index_col=0)
    print(df.head(10), f"\n{'-'*25}")
    if col == None:
        return
    print("Null Hypothesis: UAC's mean fall rate is greater than or equal to DAC's")
    print("Alternate Hypothesis: UAC's mean fall rate is strictly less than DAC's")    
    print(f"{'-'*25}")
    for f in [100, 120]:
        print(f"Test Column: {col} - Force {f} \n{'-'*25}")
        dac_flat = df[col][(df['Arch.'] == 'DAC') & (df['Terrain'] == 'Flat') & (df['Force'] == f)].to_numpy()
        uac_flat = df[col][(df['Arch.'] == 'UAC') & (df['Terrain'] == 'Flat') & (df['Force'] == f)].to_numpy()
        t_stat, p_val = stats.ttest_ind(uac_flat, dac_flat, equal_var=False, alternative='less')
        print(f"t-value: {t_stat:.5f}")
        print(f"P-value: {p_val:.5f}")
        if p_val < 0.05:
            print("Reject the Null! UAC has a statistically significant, lower fall rate than DAC on Flat terrain.")
        else:
            print("Fail to reject the Null. We cannot prove UAC has a lower fall rate than DAC on Flat terrain.")
        print(f"{'-'*25}")
        dac_flat = df[col][(df['Arch.'] == 'DAC') & (df['Terrain'] == 'Stair') & (df['Force'] == f)].to_numpy()
        uac_flat = df[col][(df['Arch.'] == 'UAC') & (df['Terrain'] == 'Stair') & (df['Force'] == f)].to_numpy()
        t_stat, p_val = stats.ttest_ind(uac_flat, dac_flat, equal_var=False, alternative='less')
        print(f"t-value: {t_stat:.5f}")
        print(f"P-value: {p_val:.5f}")
        if p_val < 0.05:
            print("Reject the Null! UAC has a statistically significant, lower fall rate than DAC on Stair terrain.")
        else:
            print("Fail to reject the Null. We cannot prove UAC has a lower fall rate than DAC on Stair terrain.")
        print(f"\n{'-'*25}")

push('Fall Rate')





def inference(col=None):
    df = pd.read_csv("dataset/inference.csv", index_col=0)
    print(df.head(10), f"\n{'-'*25}")
    if col is None:
        return
    print("Test Column: ", col)
    # Dynamically configure test logic depending on the chosen hardware metric
    if col == 'Throughput (Hz)':
        alt_direction = 'greater'
        null_msg = "UAC's mean throughput is less than or equal to DAC's (UAC is slower or equal)"
        alt_msg = "UAC's mean throughput is strictly greater than DAC's (UAC is faster)"
        success_msg = "Reject the Null! UAC achieves a statistically significant, higher throughput than DAC"
        fail_msg = "Fail to reject the Null. We cannot prove UAC achieves a higher throughput than DAC"
    else:
        # Handles lower-is-better metrics: 'Mean Latency (ms)', 'P95 (ms)', or 'Average Power Draw (W)'
        alt_direction = 'less'
        null_msg = f"UAC's mean {col} is greater than or equal to DAC's (UAC is worse or equal)"
        alt_msg = f"UAC's mean {col} is strictly less than DAC's (UAC is better/lower)"
        success_msg = f"Reject the Null! UAC has a statistically significant, lower {col} than DAC"
        fail_msg = f"Fail to reject the Null. We cannot prove UAC has a lower {col} than DAC"
    print("Null Hypothesis: ", null_msg)
    print("Alternate Hypothesis: ", alt_msg)
    print(f"{'-'*25}")
    for terrain in ['Flat', 'Stair']:
        dac = df[col][(df['Arch.'] == 'DAC') & (df['Terrain'] == terrain)].to_numpy()
        uac = df[col][(df['Arch.'] == 'UAC') & (df['Terrain'] == terrain)].to_numpy()
        t_stat, p_val = stats.ttest_ind(uac, dac, equal_var=False, alternative=alt_direction)
        print(f"--- Terrain: {terrain} ---")
        print(f"t-value: {t_stat:.5f}")
        print(f"P-value: {p_val:.5f}")
        if p_val < 0.05:
            print(f"{success_msg} on {terrain} terrain.")
        else:
            print(f"{fail_msg} on {terrain} terrain.")
        print(f"{'-'*25}")

inference('Throughput (Hz)')     
inference('Mean Latency (ms)')
inference('Average Power Draw (W)')





def cl(col=None):
    df = pd.read_csv("dataset/cl.csv", index_col=0)
    print(df.head(10), f"\n{'-'*25}")
    if col is None:
        return
    print("Test Column: ", col)
    # Configure statistical directions and descriptions based on metric type
    if col == 'Mean reward':
        alt_direction = 'greater'
        null_msg = "UAC is worse than or equal to DAC in reward"
        alt_msg = "UAC is better than DAC in reward"
        success_msg_flat = "Reject the Null! UAC obtains significantly higher reward than DAC on Slope terrain."
        fail_msg_flat = "Fail to reject the Null. We cannot prove UAC obtains significantly higher reward than DAC on Slope terrain."
        success_msg_stair = "Reject the Null! UAC obtains significantly higher reward than DAC on Stair terrain."
        fail_msg_stair = "Fail to reject the Null. We cannot prove UAC obtains significantly higher reward than DAC on Box terrain."
    else:
        # Handles lower-is-better metrics: 'Training time (min)', 'Total energy (J)', 'Peak temp (°C)', 'Peak process RAM (GiB)'
        alt_direction = 'less'
        null_msg = f"UAC uses more or equal {col} compared to DAC"
        alt_msg = f"UAC uses strictly less {col} compared to DAC"
        success_msg_flat = f"Reject the Null! UAC uses significantly less {col} than DAC on Slope terrain."
        fail_msg_flat = f"Fail to reject the Null. We cannot prove UAC uses less {col} than DAC on Slope terrain."
        success_msg_stair = f"Reject the Null! UAC uses significantly less {col} than DAC on Box terrain."
        fail_msg_stair = f"Fail to reject the Null. We cannot prove UAC uses less {col} than DAC on Box terrain."
    print("Null Hypothesis: ", null_msg)
    print("Alternate Hypothesis: ", alt_msg)
    print(f"{'-'*25}")
    # --- Slope Terrain Block (Stored in your flat variables) ---
    dac_flat = df[col][(df['Architecture'] == 'DAC') & (df['Terrain'] == 'slope')].to_numpy()
    uac_flat = df[col][(df['Architecture'] == 'UAC') & (df['Terrain'] == 'slope')].to_numpy()
    t_stat, p_val = stats.ttest_ind(uac_flat, dac_flat, equal_var=False, alternative=alt_direction)
    print(f"t-value: {t_stat:.5f}")
    print(f"P-value: {p_val:.5f}")
    if p_val < 0.05:
        print(success_msg_flat)
    else:
        print(fail_msg_flat)
    print(f"{'-'*25}")
    # --- Box Terrain Block ---
    dac_stair = df[col][(df['Architecture'] == 'DAC') & (df['Terrain'] == 'box')].to_numpy()
    uac_stair = df[col][(df['Architecture'] == 'UAC') & (df['Terrain'] == 'box')].to_numpy()
    t_stat, p_val = stats.ttest_ind(uac_stair, dac_stair, equal_var=False, alternative=alt_direction)
    print(f"t-value: {t_stat:.5f}")
    print(f"P-value: {p_val:.5f}")
    if p_val < 0.05:
        print(success_msg_stair)
    else:
        print(fail_msg_stair)
    print(f"{'-'*25}")

cl('Total energy (J)')
cl('Peak temp (°C)')
cl('Training time (min)')
cl('Mean reward')





def forgetting(col='Total_Reward'):
    df = pd.read_csv("dataset/forgetting.csv", index_col=0)
    print(df.head(10), f"\n{'-'*25}")
    if col is None:
        return
    print("Test Column: ", col)
    print("Null Hypothesis: UAC's mean performance drop is >= DAC's mean drop (UAC forgets MORE or equal)")
    print("Alternate Hypothesis: UAC's mean performance drop is < DAC's mean drop (UAC forgets LESS)")
    print(f"{'-'*25}")
    for terrain in ['Flat', 'Stair']:
        # 1. Filter out pre and post arrays for DAC
        dac_pre = df[col][(df['Arch.'] == 'DAC') & (df['Terrain'] == terrain) & (df['Stage'] == 'pre')].to_numpy()
        dac_post = df[col][(df['Arch.'] == 'DAC') & (df['Terrain'] == terrain) & (df['Stage'] == 'post')].to_numpy()
        # 2. Filter out pre and post arrays for UAC
        uac_pre = df[col][(df['Arch.'] == 'UAC') & (df['Terrain'] == terrain) & (df['Stage'] == 'pre')].to_numpy()
        uac_post = df[col][(df['Arch.'] == 'UAC') & (df['Terrain'] == terrain) & (df['Stage'] == 'post')].to_numpy()
        # 3. Calculate the degradation magnitude (Drop) per seed
        dac_drop = dac_pre - dac_post
        uac_drop = uac_pre - uac_post
        # 4. Correct Math: 'less' checks if UAC's drop is significantly SMALLER than DAC's drop
        t_stat, p_val = stats.ttest_ind(uac_drop, dac_drop, equal_var=False, alternative='less')
        print(f"--- Native Terrain: {terrain} ---")
        print(f"t-value: {t_stat:.5f}")
        print(f"P-value: {p_val:.5f}")       
        if p_val < 0.05:
            print(f"Reject the Null! UAC has a statistically significant advantage: it forgets LESS than DAC on {terrain}.")
        else:
            print(f"Fail to reject the Null. We cannot prove UAC forgets less than DAC on {terrain}.")
        print(f"{'-'*25}")

# Execute the calibrated ablation comparison
forgetting('Total_Reward')


