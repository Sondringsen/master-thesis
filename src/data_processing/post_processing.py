import numpy as np
import pandas as pd
from config import post_processing_config


def post_processing(df: pd.DataFrame, clip_value_max: float, clip_value_min: float, vol: float, drift: float):
    """
    Clips, scales and remove drift from generated data.

    Args: 
        df (pd.DataFrame): a dataframe with price paths.
        clip_value_max: the maximum value of log returns.
        clip_value_min: the minimum value of log returns.
        vol: the volatility of the returns.
        drift: the drift of the price paths.
    
    Returns:
        pd.DataFrame: A new dataframe where values are clipped, scaled and removed drift. 
            The new dataframe will be normalized from 1
    """
    log_returns = np.log(df/df.shift(axis=1)).dropna(axis=1)
    print(log_returns)

    # Clip
    log_returns = log_returns.clip(upper=clip_value_max, lower=clip_value_min)
    print(log_returns)
    
    # Scale
    scaling_factor = vol / log_returns.iloc[:, -1].std()
    log_returns *= scaling_factor

    # Drift
    drift = np.mean(log_returns, axis=0)
    log_returns = log_returns - drift.values
    

    # Price paths
    price_paths = log_returns.cumsum(axis=1)
    

    return price_paths


if __name__ == "__main__":
    print(post_processing_config)
    df = pd.read_csv("data/processed/gbm_synth_data.csv", index_col=0)
    pdf = 400*np.exp(post_processing(df, **post_processing_config))

    import matplotlib.pyplot as plt

    # Plot the first 30 price paths
    plt.figure(figsize=(12, 6))
    for i in range(min(30, len(pdf))):
        plt.plot(pdf.columns, pdf.iloc[i], color="Red", alpha=0.5)

    plt.title('First 30 Price Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.grid()
    plt.show()

