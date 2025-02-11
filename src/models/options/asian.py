from option import Option   

class Asian(Option):
    def __init__(self, strike_price):
        super().__init__(strike_price)

    def calculate_payoff(self, time_series):
        """Calculate the payoff of an Asian call option at expiry."""
        average_price = sum(time_series) / len(time_series)
        return max(average_price - self.strike_price, 0)