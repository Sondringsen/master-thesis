import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_closing_prices():
    df = pd.read_csv("data/raw/spy_daily_closing_prices.csv", index_col=0)
    df.index = pd.to_datetime(df.index)

    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df.Close)
    plt.title("SPY closing prices")
    plt.savefig("results/plots/spy_daily_closing_prices.png")

    df["log_returns"] = np.log(df["Close"]/df["Close"].shift(1))
    plt.figure(figsize=(12, 8))
    plt.hist(df.log_returns)
    plt.title("Histogram of daily SPY log returns")
    plt.savefig("results/plots/spy_daily_log_returns.png")

def describe_data():
    df = pd.read_csv("data/raw/spy_daily_closing_prices.csv", index_col=0)
    df["Log Returns"] = np.log(df["Close"]/df["Close"].shift(1))

    description = df.describe()
    with open("results/tables/spy_daily_closing_prices_describe.tex", "w") as f:
        f.write(description.to_latex
                (
                caption="Description of data.",
                label="tab:data_desc",
            )
        )


def main():
    visualize_closing_prices()
    describe_data()

if __name__ == "__main__":
    main()