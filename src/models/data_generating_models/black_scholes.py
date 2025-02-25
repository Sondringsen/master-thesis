import numpy as np
import pandas as pd
from data_generating_model import DataGeneratingModel

class BlackScholes(DataGeneratingModel):
    def __init__(self, S0, mu, sigma, T, N, M):
        """Initializes the Black-Scholes price paths model.
        
        Args:
            S0 (float): The initial price of the asset.
            mu (float): The drift of the asset.
            sigma (float): The volatility of the asset.
            T (float): The time to maturity.
            N (int): The number of time steps.
            M (int): The number of paths to generate.
        """
        self.S0 = S0
        self.sigma = sigma
        self.mu = mu
        self.T = T
        self.N = N
        self.M = M
        self.dt = T / N

    def generate_data(self):
        """Generates M price paths for the Black-Scholes model. This process follows a GBM."""
        Z = np.random.normal(size=(self.M, self.N))
        price_paths = self.S0 * np.exp(np.cumsum((self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * Z, axis=1))
        time = np.linspace(0, self.T, self.N)
        price_paths = pd.DataFrame(price_paths.T, index=time)

        return price_paths
    
    def save_data(self, path="data/processed/BS_price_paths.csv"):
        """Saves the price paths to a file."""
        price_paths = self.generate_data()
        price_paths.to_csv(path, index=True)


if __name__ == "__main__":
    option_config = {
        "S": 100,
        "K": 100,
        "mu": 0.1,
        "sigma": 0.2,
        "T": 1,
        "r": 0.05
    }

    S0 = option_config["S"]
    mu = option_config["mu"]
    sigma = option_config["sigma"]
    T = option_config["T"]
    r = option_config["r"]
    N = 10
    M = 20

    BS = BlackScholes(S0, mu, sigma, T, N, M)
    BS.save_data("path")



    

