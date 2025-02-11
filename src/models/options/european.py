from option import Option

class European(Option):
    def __init__(self, strike_price):
        super().__init__(strike_price)
        
    def calculate_payoff(self, time_series):
        """Calculate the payoff of a European call option at expiry."""
        return max(time_series[-1] - self.strike_price, 0)
    