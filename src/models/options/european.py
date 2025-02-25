import numpy as np
from options.option import Option
from scipy.stats import norm

class European(Option):
    def __init__(self, strike_price):
        super().__init__(strike_price)
        
    def calculate_payoff(self, data):
        """Calculate the payoff of a European call option at expiry."""
        return np.max(data - self.strike_price, 0)