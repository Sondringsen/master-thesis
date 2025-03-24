import numpy as np
import pandas as pd

df = pd.read_csv("data/raw/spy_daily_closing_prices.csv", index_col=0)
vol = df.loc[:, "vol"].iloc[-30:].mean()/100

post_processing_config = {
    "clip_value_max": np.log(1.2/1),
    "clip_value_min": np.log(0.8/1), # circuit breaker for sp500
    "vol": vol,
    "drift": 0

}