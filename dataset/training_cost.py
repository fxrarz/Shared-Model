import pandas as pd

data = {
  "Terrain": ["Flat", "Flat", "Flat", "Flat", "Flat", "Flat", "Flat", "Flat", "Flat", "Flat", "Stair", "Stair", "Stair", "Stair", "Stair", "Stair", "Stair", "Stair", "Stair", "Stair"],
  "Arch.": ["DAC", "DAC", "DAC", "DAC", "DAC", "UAC", "UAC", "UAC", "UAC", "UAC", "DAC", "DAC", "DAC", "DAC", "DAC", "UAC", "UAC", "UAC", "UAC", "UAC"],
  "Seed": ["42", "123", "456", "789", "999", "42", "123", "456", "789", "999", "42", "123", "456", "789", "999", "42", "123", "456", "789", "999"],
  "Run": ["[ala5prnd](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/ala5prnd)", "[d50lwydl](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/d50lwydl)", "[07wbk0ri](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/07wbk0ri)", "[96mdzl30](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/96mdzl30)", "[hf65nw84](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/hf65nw84)", "[54ntbprk](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/54ntbprk)", "[23zp55od](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/23zp55od)", "[ojfjurhs](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/ojfjurhs)", "[i9kj5fae](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/i9kj5fae)", "[oik21570](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/oik21570)", "[q3t3tk6u](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/q3t3tk6u)", "[1al47x6d](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/1al47x6d)", "[vnsyg0p6](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/vnsyg0p6)", "[76uei17p](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/76uei17p)", "[9oom87oi](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/9oom87oi)", "[gnuq8f1c](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/gnuq8f1c)", "[ovc71d55](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/ovc71d55)", "[jo1lficy](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/jo1lficy)", "[cu0nhgvc](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/cu0nhgvc)", "[towdqewn](https://wandb.ai/my-phd-research/Shared-Arch-v1/runs/towdqewn)"],
  "Training time": ["18.40 min", "18.43 min", "18.33 min", "18.35 min", "17.92 min", "15.88 min", "15.82 min", "15.65 min", "15.88 min", "15.90 min", "67.17 min", "66.87 min", "69.13 min", "71.38 min", "70.95 min", "61.43 min", "60.98 min", "61.37 min", "59.33 min", "61.02 min"],
  "Peak GPU VRAM": ["50.71%", "50.71%", "50.71%", "50.71%", "50.71%", "49.63%", "49.61%", "49.63%", "49.63%", "49.63%", "57.84%", "57.06%", "58.26%", "57.06%", "57.84%", "57.94%", "57.16%", "58.35%", "57.16%", "57.94%"],
  "Peak process RAM": ["4.82 GiB", "4.81 GiB", "4.81 GiB", "4.83 GiB", "4.81 GiB", "4.81 GiB", "4.84 GiB", "4.83 GiB", "4.82 GiB", "4.81 GiB", "4.97 GiB", "4.96 GiB", "4.97 GiB", "4.96 GiB", "4.98 GiB", "4.96 GiB", "4.99 GiB", "4.97 GiB", "4.96 GiB", "4.97 GiB"],
  "Peak host RAM": ["62.01%", "62.70%", "62.70%", "62.46%", "62.88%", "62.91%", "62.58%", "62.92%", "63.47%", "63.33%", "61.16%", "62.16%", "64.14%", "68.09%", "64.14%", "65.92%", "68.47%", "67.89%", "63.45%", "63.61%"]
}

df = pd.DataFrame(data)
df['Training time'] = df['Training time'].str.replace(' min', '', regex=False)
df['Peak GPU VRAM'] = df['Peak GPU VRAM'].str.replace('%', '', regex=False)
df['Peak process RAM'] = df['Peak process RAM'].str.replace(' GiB', '', regex=False)
df['Peak host RAM'] = df['Peak host RAM'].str.replace('%', '', regex=False)

cols_to_convert = ['Training time', 'Peak GPU VRAM', 'Peak process RAM', 'Peak host RAM']

for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.rename(columns={'Training time': 'Training time (min)'}, inplace=True)
df.rename(columns={'Peak GPU VRAM': 'Peak GPU VRAM (%)'}, inplace=True)
df.rename(columns={'Peak process RAM': 'Peak process RAM (GiB)'}, inplace=True)
df.rename(columns={'Peak host RAM': 'Peak host RAM (%)'}, inplace=True)

df.to_csv("training_cost.csv")
