import numpy as np
import pandas as pd
# from models.data_generating_models.data_generating_model import DataGeneratingModel

from abc import abstractmethod, ABC
import pandas as pd
import numpy as np
from scipy.integrate import simpson
import scipy.stats as stats

class DataGeneratingModel(ABC):
    def __init__(self, train_data: pd.DataFrame, N: int, M: int, load_params: bool, config: dict = None):
        self.train_data = train_data
        try:
            self.train_data_returns = train_data.pct_change().dropna().to_numpy().flatten()
        except:
            pass
        self.N = N
        self.M = M
        self.load_params = load_params
        self.synth_data = None
        self.config = config

    @abstractmethod
    def fit_params_to_data(self):
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def generate_data(self, save: bool = False):
        raise NotImplementedError("Subclasses should implement this method")  

    @abstractmethod
    def _objective(self) -> float:
        raise NotImplementedError("Subclasses should implement this method")


    @abstractmethod
    def _save_params(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    @abstractmethod
    def _load_params(self):
        raise NotImplementedError("Subclasses should implement this method")


    def _compute_kde(self, data, grid):
        kde = stats.gaussian_kde(data)
        return kde(grid)

    def _kl_divergence(self, p, q):
        p = np.maximum(p, 1e-8)  # Avoid log(0)
        q = np.maximum(q, 1e-8)
        return simpson(p * np.log(p / q), dx=0.01)
    

    def _save_synth_data(self, path: str):
        # Should maybe have a data validation here
        self.synth_data.to_csv(path)

class SchrodingerBridge(DataGeneratingModel):
    def __init__(self, train_data: pd.DataFrame, N: int, M: int, load_params: bool, config: dict):
        super().__init__(train_data, N, M, load_params, config)
        # Create sequences of 30 days from the price data
        prices = self.train_data.iloc[:, 0].values
        # Standardize prices to have mean 0 and unit variance
        prices = (prices - np.mean(prices)) / np.std(prices)
        self.X = np.array([prices[i:i+M] for i in range(len(prices)-M + 1)])
        
    def fit_params_to_data(self):

        pass


    def _F_i(self, t: float, t_i: float, t_p_1: float, x_i: float, x: float, x_p_1: float) -> float:
        return np.exp(np.linalg.norm(x_p_1 - x)/(2*(t_p_1 - t)) + np.linalg.norm(x_p_1 - - x_i)/(2*(t_p_1 - t_i)))
    
    def _quartic_kernel(self, x: float, h: float) -> float:
        """Quartic kernel (biweight kernel)"""
        norm = np.linalg.norm(x / h)
        return (1 - norm**2) * (norm <= 1)
    
    def _kernel_estimator(self, t: float, x: float, x_i: np.array, t_i: float, t_p_1: float, i: int, i_p_1: int):
        h = self.config["h"]
        M = self.X.shape[0] # this is the same M as in the paper, but not the same M as in the DataGeneratingModel class
        i = len(x_i)
        numerator = 0
        denominator = 0

        for m in range(M):
            sample = self.X[m]
            x_m_ti = sample[i]
            x_m_tip1 = sample[i_p_1]
            Fi_val = self._F_i(t, t_i, t_p_1, x_m_ti, x, x_m_tip1)
            
            # Product of kernels for all xj - xj_m
            kernel_prod = np.prod([
                self._quartic_kernel(x_i[j] - sample[j], h) for j in range(i)
            ]) if i > 0 else 1.0

            weight = Fi_val * kernel_prod
            numerator += (x_m_tip1 - x) * weight
            denominator += weight
        if denominator == 0:
            return np.zeros_like(x)

        drift = numerator / ((t_p_1 - t) * denominator)
        return drift
        

    def generate_data(self):
        N_pi = self.config["N_pi"]
        x = np.zeros((self.N, self.M))
        epsilon = np.random.normal(0, 1, (self.N, self.M - 1, N_pi - 1))
        for j in range(self.N):
            t = np.linspace(0, 1, self.M)
            print(f"Generating time series {j}...")
            for i in range(self.M - 1):
                y = np.zeros(N_pi)
                for k in range(N_pi - 1):
                    t_k_i_pi = t[i] + k/N_pi
                    a_hat = self._kernel_estimator(t_k_i_pi, y[k], x[j, :i], t[i], t[i+1], i, i+1)
                    y[k+1] = y[k] + 1/N_pi*a_hat + 1/np.sqrt(N_pi)*epsilon[j, i, k]
                x[j, i+1] = y[-1]
        self.synth_data = x

            

    def _objective(self):
        pass

    def _load_params(self):
        pass

    def _save_params(self):
        pass


if __name__ == "__main__":
    hyperparameters = {
        "h": 0.05,
        "N_pi": 100,
    }
    model = SchrodingerBridge(train_data=pd.read_csv("data/raw/spy_daily_closing_prices.csv", index_col=0), N=1, M=30, load_params=False, config=hyperparameters)
    model.generate_data()
    print(model.synth_data)
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,6))
    for i in range(model.synth_data.shape[0]):
        plt.plot(model.synth_data[i,2:], alpha=0.3)
    plt.title('Generated Synthetic Paths')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()