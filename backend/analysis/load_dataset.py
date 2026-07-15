import pandas as pd

# load dataset
df = pd.read_csv("data/raw/jobs_dataset.csv")

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nSample Data:")
print(df.head())