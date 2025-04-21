import numpy as np
import pandas as pd
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "../../data/raw/spy_daily_closing_prices_train.csv")
df = pd.read_csv(file_path, index_col=0)
# df = pd.read_csv("data/raw/spy_daily_closing_prices.csv", index_col=0)
vol = df.loc[:, "vol"].iloc[-30:].mean()/100
s0 = df.loc[:, "Close"].iloc[-1]

post_processing_config = {
    "clip_value_max": np.log(1.2/1),
    "clip_value_min": np.log(0.8/1), # circuit breaker for sp500
    "vol": vol,
    "drift": 0,
    "s0": s0
}