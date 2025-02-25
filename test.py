import pandas as pd

df = pd.read_csv("data/processed/BS_price_paths.csv", index_col=0)

returns = df.shift(-1)-df
print(returns)
