import numpy as np
import pandas as pd
from scipy.stats import norm
from hedging_agent import HedgingAgent

class DeltaHedging(HedgingAgent):
    def __init__(self, option, data, K, r):
        super().__init__(option, data, K, r)
        

    def get_action(self):
        t = 1 - self.data.index
        t = np.tile(t,  (self.data.shape[1], 1)).T
        est_vol = np.tile(self.data.std(axis=1), (self.data.shape[1], 1)).T
        d1 = (np.log(self.data / self.K) + (self.r + est_vol **2 / 2) * t) / (est_vol * np.sqrt(t))
        delta = -norm.cdf(d1)
        self.actions = delta
    
    def loss(self):
        returns = self.actions[:-1, :] * ((self.data.shift(-1) - self.data).dropna()) - self.option.calculate_returns(data)
        # print(self.actions[:-1, :] * ((self.data.shift(-1) - self.data).dropna()))
        # print()
        # print(self.option.calculate_returns(data))
        return np.mean((returns-0)**2)
    

if __name__ == "__main__":
    option_config = {
        "S": 100,
        "K": 100,
        "mu": 0.1,
        "sigma": 0.2,
        "T": 1,
        "r": 0.05
    }
    from options.european import European
    data = pd.read_csv("data/processed/BS_price_paths.csv", index_col=0)
    option = European(option_config["K"])
    ha = DeltaHedging(option, data, option_config["K"], option_config["r"])
    ha.get_action()
    loss = ha.loss()
    print(loss)
    

    

    
