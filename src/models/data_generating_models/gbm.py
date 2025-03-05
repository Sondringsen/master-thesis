import numpy as np
import pandas as pd
from models.data_generating_models.data_generating_model import DataGeneratingModel
from scipy.optimize import minimize
import pickle

class GBM(DataGeneratingModel):
    def __init__(self, train_data: pd.DataFrame, N: int, M: int, **params):
        """Initializes the Black-Scholes price paths model.
        
        Args:
            train_data (pd.DataFrame): The training data to fit the model to.
            N (int): The number of time steps.
            M (int): The number of paths to generate.
            **params (dict): The parameters of the GBM model.
        """
        if not params:
            params = {
                "mu": None,
                "sigma": None,
            }
        else:
            params = params["params"]
        print(params)
        super().__init__(train_data, N, M, **params)

    def loss_func(self, params):
        """The negative log-likelihood function for minimization."""
        dt = self.N/252

        mu, sigma = params
        returns = np.diff(np.log(self.train_data["Close"]))
        n = len(returns)
        log_likelihood = -n/2 * np.log(2 * np.pi * sigma**2 * dt) - 1/(2 * sigma**2 * dt) * np.sum((returns - mu * dt)**2)
        return -log_likelihood

    def fit_params_to_data(self):
        """Fits the parameters of the Black-Scholes model to the data."""
        initial_guess = [0.1, 0.2]
        bounds = [(None, None), (1e-6, None)]
        result = minimize(self.loss_func, initial_guess, bounds=bounds)
        mu_tilde, sigma_tilde = result.x
        
        param_dic = {"mu": mu_tilde, "sigma": sigma_tilde}
        param_file = open('data/params/gbm_params.pkl', 'wb')
        pickle.dump(param_dic, param_file)
        param_file.close()

    def generate_data(self):
        """Generates M price paths with N timesteps using a GBM model."""
        Z = np.random.normal(size=(self.M, self.N))

        #### TODO: S0 CANNOT BE 100 HERE, MUST SET IT SOMEWHERE ELSE
        price_paths = 100 * np.exp(np.cumsum((self.params["mu"] - 0.5 * self.params["sigma"] ** 2) * self.N/252 + self.params["sigma"] * np.sqrt(self.N/252) * Z, axis=1))
        self.synth_data = pd.DataFrame(price_paths)

        file_path = "data/processed/gbm_synth_data.csv"
        self._save_synth_data(file_path)

if __name__ == "__main__":
    train_data = pd.read_csv("data/raw/spy_daily_closing_prices.csv", index_col=0)
    params = pd.read_csv("data/params/gbm_params.csv", index_col=0)
    print(params)
    # print(train_data.head())
    # N = 30
    # M = 1000
    # gbm = GBM(train_data, N, M)
    # gbm.fit_params_to_data()


    # def negative_log_likelihood(params, data, dt):
    #     mu, sigma = params
    #     returns = np.diff(np.log(data))
    #     n = len(returns)
    #     log_likelihood = -n/2 * np.log(2 * np.pi * sigma**2 * dt) - 1/(2 * sigma**2 * dt) * np.sum((returns - mu * dt)**2)
    #     return -log_likelihood
    
    # bounds = [(None, None), (1e-6, None)]
    
    # opt = minimize(negative_log_likelihood, [0.1, 0.2], args=(train_data, N/252), bounds=bounds)
    # print(opt.x)
    # print(negative_log_likelihood([100, 0.01], train_data, N/252, bounds=bounds))




    

