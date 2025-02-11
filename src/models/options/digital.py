from option import Option

class Digital(Option):
    def __init__(self, strike_price):
        super().__init__(strike_price)

    def calculate_payoff(self, time_series):
        """Calculate the payoff of a digital option at expiry."""
        if time_series[-1] > self.strike_price:
            return 1
        else:
            return 0