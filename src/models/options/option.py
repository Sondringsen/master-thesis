class Option:
    def __init__(self, strike_price):
        self.strike_price = strike_price

    def calculate_payoff(self, time_series):
        raise NotImplementedError("Subclasses should implement this method")