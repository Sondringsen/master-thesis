import numpy as np
import pandas as pd



def post_processing(df: pd.DataFrame, clip_value_max: float, clip_value_min: float, vol: float, drift: float, s0: float):
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
    # Ensure all positive values
    df = df + np.abs(df.min()) + 1

    # Log returns
    log_returns = np.log(df/df.shift(axis=1)).dropna(axis=1)

    # Clip
    log_returns = log_returns.clip(upper=clip_value_max, lower=clip_value_min)
    
    # Scale
    # scaling_factor = 2*vol / log_returns.iloc[:, -1].std()
    scaling_factor = vol/log_returns.std()
    log_returns *= scaling_factor

    # Drift
    drift = np.mean(log_returns, axis=0)
    log_returns = log_returns - drift.values

    # Price paths
    price_paths = log_returns.cumsum(axis=1)
    price_paths = np.exp(price_paths)
    ones = pd.DataFrame(np.ones(price_paths.shape[0]))
    price_paths = pd.DataFrame(price_paths)
    price_paths = pd.concat([ones, price_paths], axis=1)
    
    return price_paths


if __name__ == "__main__":
    from config import post_processing_config
    # df = pd.read_csv("data/processed/gbm_synth_data.csv", index_col=0)
    # pdf = post_processing(df, **post_processing_config)

    # import matplotlib.pyplot as plt

    # # Plot the first 30 price paths
    # plt.figure(figsize=(12, 6))
    # for i in range(min(30, len(pdf))):
    #     plt.plot(pdf.columns, pdf.iloc[i], color="Red", alpha=0.5)

    # plt.title('First 30 Price Paths')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Price')
    # plt.grid()
    # plt.show()

    # Generate random data and test post-processing
    import numpy as np
    import pandas as pd
    
    # Generate random matrix
    random_data = np.random.randn(100000, 30)
    df = pd.DataFrame(random_data)
    
    # Apply post-processing with default parameters
    processed_df = post_processing(df, **post_processing_config)
    
    print(f"Shape of processed data: {processed_df.shape}")
    print("\nFirst few rows:")
    print(processed_df.head())

