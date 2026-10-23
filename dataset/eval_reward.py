import pandas as pd

data = {
  "Terrain": ["Flat", "Flat", "Flat", "Flat", "Flat", "Flat", "Flat", "Flat", "Flat", "Flat", "Stair", "Stair", "Stair", "Stair", "Stair", "Stair", "Stair", "Stair", "Stair", "Stair"],
  "Arch.": ["DAC", "DAC", "DAC", "DAC", "DAC", "UAC", "UAC", "UAC", "UAC", "UAC", "DAC", "DAC", "DAC", "DAC", "DAC", "UAC", "UAC", "UAC", "UAC", "UAC"],
  "Seed": [42, 123, 456, 789, 999, 42, 123, 456, 789, 999, 42, 123, 456, 789, 999, 42, 123, 456, 789, 999],
  "Reward": [37.321, 34.466, 37.865, 39.117, 37.578, 38.022, 36.439, 38.474, 39.091, 38.725, 27.064, 29.387, 32.886, 30.344, 30.763, 18.798, 26.606, 26.835, 26.471, 26.822]
}

df = pd.DataFrame(data)
df.to_csv("eval_reward.csv")
