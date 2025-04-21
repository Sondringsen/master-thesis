import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.manifold import TSNE

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


def avg_over_dim(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Average over the feature dimension of the data.

    Args:
        data (np.ndarray): The data to average over.
        axis (int): Axis to average over.

    Returns:
        np.ndarray: The data averaged over the feature dimension.
    """
    return np.mean(data, axis=axis)


def visualize_and_save_tsne(
    samples1: np.ndarray,
    samples1_name: str,
    samples2: np.ndarray,
    samples2_name: str,
    scenario_name: str,
    save_dir: str,
    max_samples: int = 1000,
) -> None:
    """
    Visualize the t-SNE of two sets of samples and save to file.

    Args:
        samples1 (np.ndarray): The first set of samples to plot.
        samples1_name (str): The name for the first set of samples in the plot title.
        samples2 (np.ndarray): The second set of samples to plot.
        samples2_name (str): The name for the second set of samples in the
                             plot title.
        scenario_name (str): The scenario name for the given samples.
        save_dir (str): Dir path to which to save the file.
        max_samples (int): Maximum number of samples to use in the plot. Samples should
                           be limited because t-SNE is O(n^2).
    """
    if samples1.shape != samples2.shape:
        raise ValueError(
            "Given pairs of samples dont match in shapes. Cannot create t-SNE.\n"
            f"sample1 shape: {samples1.shape}; sample2 shape: {samples2.shape}"
        )

    # samples1_2d = avg_over_dim(samples1, axis=2)
    # samples2_2d = avg_over_dim(samples2, axis=2)

    samples1_2d = samples1
    samples2_2d = samples2

    # num of samples used in the t-SNE plot
    used_samples = min(samples1_2d.shape[0], max_samples)

    # Combine the original and generated samples
    combined_samples = np.vstack(
        [samples1_2d[:used_samples], samples2_2d[:used_samples]]
    )

    # Compute the t-SNE of the combined samples
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300, random_state=42)
    tsne_samples = tsne.fit_transform(combined_samples)

    # Create a DataFrame for the t-SNE samples
    tsne_df = pd.DataFrame(
        {
            "tsne_1": tsne_samples[:, 0],
            "tsne_2": tsne_samples[:, 1],
            "sample_type": [samples1_name] * used_samples
            + [samples2_name] * used_samples,
        }
    )

    # Plot the t-SNE samples
    plt.figure(figsize=(8, 8))
    for sample_type, color in zip([samples1_name, samples2_name], ["red", "blue"]):
        if sample_type is not None:
            indices = tsne_df["sample_type"] == sample_type
            plt.scatter(
                tsne_df.loc[indices, "tsne_1"],
                tsne_df.loc[indices, "tsne_2"],
                label=sample_type,
                color=color,
                alpha=0.5,
                s=100,
            )

    plt.title(f"t-SNE for {scenario_name}")
    plt.legend()

    # Save the plot to a file
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{scenario_name}.png"))

    plt.show()


def main():
    # visualize_data()
    # describe_data()
    true = pd.read_csv("data/raw/spy_daily_closing_prices.csv", index_col=0)[["Close"]].values[:1028,0:]
    sequence_length = 29
    true_sequences = np.array([
        true[i:i + sequence_length].flatten()
        for i in range(len(true) - sequence_length + 1)
    ])
    fake = pd.read_csv("data/processed/tc_vae_synth_data.csv", index_col=0).values#.mean(axis=1, keepdims=True)

    visualize_and_save_tsne(
        true_sequences,
        "true",
        fake,
        "fake",
        "tc_vae",
        "results/plots",
        max_samples=1000
    )

if __name__ == "__main__":
    main()