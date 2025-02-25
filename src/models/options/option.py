from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import norm


class Option(ABC):
    def __init__(self, strike_price):
        self.strike_price = strike_price

    @abstractmethod
    def calculate_payoff(self, data: pd.DataFrame):
        raise NotImplementedError("Subclasses should implement this method")
    
    def _calculate_price(self, data: pd.DataFrame) -> pd.DataFrame:
        t = 1 - data.index
        t = np.tile(t,  (data.shape[1], 1)).T
        K = 100
        sigma= 0.2
        r = 0.05
        d1 = (np.log(data/K) + (r + sigma**2/2)*t)/(sigma*t**0.5)
        d2 = d1 - sigma*t**0.5

        price = norm.cdf(d1)*data - norm.cdf(d2)*K*np.exp(-r*t)
        return price
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        price = self._calculate_price(data)
        return (price.shift(-1) - price).dropna()
    
    