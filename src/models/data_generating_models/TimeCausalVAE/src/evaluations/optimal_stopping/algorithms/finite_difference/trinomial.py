from math import exp, sqrt
import numpy as np
import tqdm
from evaluations.optimal_stopping.algorithms.finite_difference import finite_difference_pricer


class TrinomialPricer(finite_difference_pricer.Finite_Difference_Pricer):
    def __init__(self, model, payoff, **kwargs):
        super().__init__(model, payoff, **kwargs)

    def price(self, stock_paths=None, verbose=1, **kwargs):
        self.set_vol_and_div()
        nb_dates = self.model.nb_dates
        deltaT = self.model.maturity / self.model.nb_dates
        discount_factor = exp(-self.model.rate * deltaT)
        up = exp(self.vol * sqrt(2 * deltaT))
        down = 1 / up
        denominator = exp(self.vol * sqrt(deltaT / 2)) - exp(-self.vol * sqrt(deltaT / 2))
        proba_up = (
            (exp((self.model.rate - self.dividend) * deltaT / 2) - exp(-self.vol * sqrt(deltaT / 2))) / denominator
        ) ** 2
        proba_down = (
            (exp(self.vol * sqrt(deltaT / 2)) - exp((self.model.rate - self.dividend) * deltaT / 2)) / denominator
        ) ** 2
        proba_middle = 1 - proba_up - proba_down

        # NEW (FAST) VERSION
        steps = range(self.model.nb_dates)
        i = np.arange(nb_dates - 1, -nb_dates, -1)
        spot_prices = self.model.spot * up**i
        option_prices = self.payoff(spot_prices)
        for n in tqdm.tqdm(reversed(steps[:-1]), disable=(not verbose)):
            spot_prices = spot_prices[1:-1]
            option_prices = discount_factor * (
                proba_up * option_prices[:-2] + proba_middle * option_prices[1:-1] + proba_down * option_prices[2:]
            )
            exercise = self.payoff(spot_prices)
            which = option_prices < exercise
            option_prices[which] = exercise[which]
        return option_prices[0], 0
