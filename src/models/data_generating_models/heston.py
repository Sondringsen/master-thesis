import numpy as np

class Heston():
    def __init__(self, S0, mu, kappa, theta, sigma, rho, v0, T, N, M):
        """Initializes the Heston price paths model.

        Args:
            S0 (float): The initial price of the asset.
            mu (float): The drift of the asset.
            kappa (float): The mean-reversion rate.
            theta (float): The long-term variance.
            sigma (float): The volatility of volatility.
            rho (float): The correlation between the asset and volatility.
            v0 (float): The initial variance.
            T (float): The time to maturity.
            N (int): The number of time steps.
            M (int): The number of paths to generate.
        """
        self.S0 = S0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.T = T
        self.N = N
        self.M = M
        self.dt = T / N

        if 2*kappa*theta <= sigma**2:
            raise ValueError("Feller condition not satisfied. Check your input parameters for the Heston model.")
        
    def generate_data(self):
        w1 = np.random.normal(size=(self.M, self.N))
        w2 = self.rho * w1 + np.sqrt(1 - self.rho ** 2) * np.random.normal(size=(self.M, self.N))

        v_t = np.zeros((self.M, self.N))
        v_t[:, 0] = self.v0
        s_t = np.zeros((self.M, self.N))
        s_t[:, 0] = self.S0

        for i in range(1, self.N):
            v_t[:, i] = (v_t[:, i - 1] + 
                        self.kappa * (self.theta - v_t[:, i - 1]) * self.dt + 
                        self.sigma * np.sqrt(v_t[:, i - 1] * self.dt) * w2[:, i-1])
            
            v_t[:, i] = np.maximum(v_t[:, i], 0)  # Ensure variance is non-negative

            s_t[:, i] = s_t[:, i-1] * np.exp((self.mu - 0.5 * v_t[:, i - 1]) * self.dt +
                                            np.sqrt(v_t[:, i - 1] * self.dt) * w1[:, i-1])

        

        return s_t

    def save_data(self, path="data/processed/Heston_price_paths.csv"):
        """Saves the price paths to a file."""
        price_paths = self.generate_paths()
        price_paths.to_csv(path, index=True)
        
