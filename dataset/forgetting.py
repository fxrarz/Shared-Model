import pandas as pd

data = {
    "Terrain": ['Flat', 'Flat', 'Flat', 'Flat', 'Stair', 'Stair', 'Stair', 'Stair', 'Flat', 'Flat', 'Flat', 'Flat', 'Stair', 'Stair', 'Stair', 'Stair', 'Flat', 'Flat', 'Flat', 'Flat', 'Stair', 'Stair', 'Stair', 'Stair', 'Flat', 'Flat', 'Flat', 'Flat', 'Stair', 'Stair', 'Stair', 'Stair', 'Flat', 'Flat', 'Flat', 'Flat', 'Stair', 'Stair', 'Stair', 'Stair'],
    "Arch.": ['DAC', 'DAC', 'UAC', 'UAC', 'DAC', 'DAC', 'UAC', 'UAC', 'DAC', 'DAC', 'UAC', 'UAC', 'DAC', 'DAC', 'UAC', 'UAC', 'DAC', 'DAC', 'UAC', 'UAC', 'DAC', 'DAC', 'UAC', 'UAC', 'DAC', 'DAC', 'UAC', 'UAC', 'DAC', 'DAC', 'UAC', 'UAC', 'DAC', 'DAC', 'UAC', 'UAC', 'DAC', 'DAC', 'UAC', 'UAC'],
    "Stage": ['pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post', 'pre', 'post'],
    "Seed": [42, 42, 42, 42, 42, 42, 42, 42, 123, 123, 123, 123, 123, 123, 123, 123, 456, 456, 456, 456, 456, 456, 456, 456, 789, 789, 789, 789, 789, 789, 789, 789, 999, 999, 999, 999, 999, 999, 999, 999],
    "Total_Reward": [
        37.320885, 36.127929, 38.021836, 36.368074, 27.063913, 7.997624, 18.798221, -0.984582,
        36.828490, 35.216494, 36.587258, 32.347470, 30.040367, 5.846555, 23.805668, -0.803185,
        36.963251, 36.870882, 37.737962, 35.926590, 31.135499, -0.294957, 26.171017, -2.470843,
        37.019595, 37.399293, 37.888288, 36.582311, 30.520633, 11.021171, 25.479473, -3.362300,
        36.961116, 36.666726, 37.606370, 36.313162, 30.168572, 5.114578, 21.675126, 23.637562
    ],
    "Terminated_At": [
        1000, 1000, 1000, 1000, 1000, 801, 1000, 228,
        1000, 1000, 1000, 1000, 1000, 482, 1000, 238,
        1000, 1000, 1000, 1000, 1000, 259, 1000, 284,
        1000, 1000, 1000, 1000, 1000, 721, 1000, 164,
        1000, 1000, 1000, 1000, 1000, 470, 1000, 1000
    ]
}

df = pd.DataFrame(data)
print(df)

df.to_csv("forgetting.csv")
