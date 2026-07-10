import pandas as pd

df = pd.read_parquet("data/predictions/predictions.parquet")

print(df.head())
print(df.shape)