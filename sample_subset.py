import json
import pandas as pd

directory = 'C:/Users/todor/OneDrive/Desktop/Proj 1 DU/RAF-PetFinder-dataset-main/Data'

train_df = pd.read_csv(f'{directory}/train.csv')
test_df = pd.read_csv(f'{directory}/test/test.csv')

print(f'Train samples: {len(train_df)}')
print(f'Test samples: {len(test_df)}')

# sample 2000 entries from train and 500 from test
train_sample = train_df.sample(n=2000, random_state=442018)
test_sample = test_df.sample(n=500, random_state=442018)

train_sample.head()