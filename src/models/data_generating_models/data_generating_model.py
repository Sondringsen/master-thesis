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
    

