import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

colors = {
    "color1":"#c49030", 
    "color2":"#6c1815"
}

def visualize_data():
    df = pd.read_csv("data/raw/spy_daily_closing_prices.csv", index_col=0)
    df.index = pd.to_datetime(df.index)

    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df.Close, color=colors["color1"])
    plt.title("SPY closing prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.savefig("results/plots/spy_daily_closing_prices.png")

    df["log_returns"] = np.log(df["Close"]/df["Close"].shift(1))
    plt.figure(figsize=(12, 8))
    plt.hist(df.log_returns, bins=30, color=colors["color1"])
    plt.title("Histogram of daily SPY log returns")
    plt.xlabel("Time")
    plt.ylabel("Log returns")
    plt.savefig("results/plots/spy_daily_log_returns.png")

    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df.vol, color=colors["color1"])
    plt.title("Volatility modeled using GJR-GARCH.")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.savefig("results/plots/gjr_garch.png")

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
    visualize_data()
    describe_data()

if __name__ == "__main__":
    main()