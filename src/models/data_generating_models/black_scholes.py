import numpy as np

class BlackScholes():
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

    def generate_paths(self):
        """Generates M price paths for the Black-Scholes model. This process follows a GBM."""
        Z = np.random.normal(size=(self.M, self.N))
        price_paths = self.S0 * np.exp(np.cumsum((self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * Z, axis=1))

        return price_paths
    

