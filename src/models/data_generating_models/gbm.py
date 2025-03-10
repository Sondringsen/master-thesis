import numpy as np
import pandas as pd
from models.data_generating_models.data_generating_model import DataGeneratingModel
from scipy.optimize import minimize
import pickle

class GBM(DataGeneratingModel):
    def __init__(self, train_data: pd.DataFrame, N: int, M: int, load_params: bool, config: dict = None):
        """Initializes the Black-Scholes price paths model.
        
        Args:
            train_data (pd.DataFrame): The training data to fit the model to.
            N (int): The number of time steps.
            M (int): The number of paths to generate.
            load_params (bool): Load parameters if set to True.
            config (dict): A dictionary of config parameters (no config for heston).
        """
        super().__init__(train_data, N, M, load_params, config)
        
        if not self.load_params:
            self.params = {
                "mu": None,
                "sigma": None,
            }
        else:
            self._load_params()


    def _objective(self, params):
        """Objective function for gbm."""
        self.params["mu"], self.params["sigma"] = params
        self.generate_data()

        train_grid = np.linspace(self.train_data_returns.min(), self.train_data_returns.max(), 1000)
        sim_grid = np.linspace(self.synth_data_returns.min(), self.synth_data_returns.max(), 1000)

        # Compute probability densities
        p_train = self._compute_kde(self.train_data_returns, train_grid)
        q_sim = self._compute_kde(self.synth_data_returns, sim_grid)
        
        return self._kl_divergence(p_train, q_sim)

    def fit_params_to_data(self):
        """Fits the parameters of the Black-Scholes model to the data."""
        initial_guess = [0.1, 0.2]
        bounds = [(None, None), (1e-6, None)]
        result = minimize(self._objective, initial_guess, bounds=bounds)
        mu_tilde, sigma_tilde = result.x
        
        param_dic = {"mu": mu_tilde, "sigma": sigma_tilde}
        self.params = param_dic

        self._save_params()

    def generate_data(self, save: bool = False):
        """Generates M price paths with N timesteps using a GBM model."""
        Z = np.random.normal(size=(self.M, self.N))
        dt = self.N/252

        #### TODO: S0 CANNOT BE 100 HERE, MUST SET IT SOMEWHERE ELSE
        S0 = 100
        price_paths = S0 * np.exp(np.cumsum((self.params["mu"] - 0.5 * self.params["sigma"] ** 2) * self.N/252 + self.params["sigma"] * np.sqrt(dt) * Z, axis=1))
        self.synth_data = pd.DataFrame(price_paths)
        self.synth_data_returns = self.synth_data.pct_change().dropna().to_numpy().flatten()

        if save:
            file_path = "data/processed/gbm_synth_data.csv"
            self._save_synth_data(file_path)

    def _save_params(self):
        with open('data/params/gbm_params.pkl', 'wb') as param_file:
            pickle.dump(self.params, param_file)

    def _load_params(self):
        with open('data/params/gbm_params.pkl', 'rb') as param_file:
            self.params = pickle.load(param_file)

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




    

