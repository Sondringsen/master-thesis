import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pickle
from models.data_generating_models.data_generating_model import DataGeneratingModel

class Heston(DataGeneratingModel):
    def __init__(self, train_data: pd.DataFrame, N: int, M: int, load_params: bool, config: dict = None):
        """Initializes the Heston price paths model.

        Args:
            train_data (pd.DataFrame): The training data to fit the model to.
            N (int): The number of time steps.
            M (int): The number of paths to generate.
            load_params (bool): Load parameters if set to True.
            config (dict): A dictionary of config parameters (no config for heston).
        """
        super().__init__(train_data, N, M, load_params, config)

        if not self.load_params:
            # self.params = {
            #     "mu": None,
            #     "v0": None,
            #     "theta": None,
            #     "kappa": None,
            #     "rho": None,
            #     "sigma": None,
            # }
            # self.params = {
            #     "mu": np.random.uniform(0, 0.01),
            #     "v0": np.random.uniform(0, 0.01),
            #     "theta": np.random.uniform(0, 0.01),
            #     "kappa": np.random.uniform(0, 0.01),
            #     "rho": np.random.uniform(0, 0.01),
            #     "sigma": np.random.uniform(0, 0.01),
            # }
            self.params = {
                "mu": 0.07,
                "v0": 0.04,
                "theta": 0.05,
                "kappa": 0.1,
                "rho": -0.4,
                "sigma": 0.2,
            }
        else:
            self._load_params()
        
        # if 2*self.params["kappa"]*self.params["theta"] <= self.params["sigma"]**2:
        #     raise ValueError("Feller condition not satisfied. Check your input parameters for the Heston model.")
        
        
        
    def _objective(self, params):
        """Objective function for heston."""
        self.params["mu"], self.params["v0"], self.params["kappa"], self.params["theta"], self.params["rho"], self.params["sigma"] = params
        self.generate_data()

        train_grid = np.linspace(self.train_data_returns.min(), self.train_data_returns.max(), 1000)
        sim_grid = np.linspace(self.synth_data_returns.min(), self.synth_data_returns.max(), 1000)

        # Compute probability densities
        p_train = self._compute_kde(self.train_data_returns, train_grid)
        q_sim = self._compute_kde(self.synth_data_returns, sim_grid)
        
        return self._kl_divergence(p_train, q_sim)

    def fit_params_to_data(self):
        """Fits the parameters of the Heston model to the data."""
        initial_guess = [0.1, 0.04, 2.0, 0.04, -0.5, 0.2]
        bounds = [(-0.1, 0.5), (0.01, 0.5), (0.1, 5.0), (0.01, 0.5), (-0.99, 0.99), (0.01, 1.0)]
        result = minimize(self._objective, initial_guess, bounds=bounds)
        mu, v0, kappa, theta, rho, sigma = result.x
        
        param_dic = {"mu": mu, "v0": v0, "kappa": kappa, "theta": theta, "rho": rho, "sigma": sigma}
        self.params = param_dic
        self._save_params()
        
        
    def generate_data(self, save: bool = False):
        "Generate M price paths with N timesteps using a Heston model."
        w1 = np.random.normal(size=(self.M, self.N))
        w2 = self.params["rho"] * w1 + np.sqrt(1 - self.params["rho"] ** 2) * np.random.normal(size=(self.M, self.N))
        dt = self.N/252

        #### TODO: S0 CANNOT BE 100 HERE, MUST SET IT SOMEWHERE ELSE
        S0 = 100

        v_t = np.zeros((self.M, self.N))
        v_t[:, 0] = self.params["v0"]
        s_t = np.zeros((self.M, self.N))
        s_t[:, 0] = S0

        for i in range(1, self.N):
            v_t[:, i] = (v_t[:, i - 1] + 
                        self.params["kappa"] * (self.params["theta"] - v_t[:, i - 1]) * dt + 
                        self.params["sigma"] * np.sqrt(v_t[:, i - 1] * dt) * w2[:, i-1])
            
            v_t[:, i] = np.maximum(v_t[:, i], 0)  # Ensure variance is non-negative

            s_t[:, i] = s_t[:, i-1] * np.exp((self.params["mu"] - 0.5 * v_t[:, i - 1]) * dt +
                                            np.sqrt(v_t[:, i - 1] * dt) * w1[:, i-1])

        
        self.synth_data = pd.DataFrame(s_t)
        self.synth_data_returns = self.synth_data.pct_change().dropna().to_numpy().flatten()
        
        if save:
            file_path = "data/processed/heston_synth_data.csv"
            self._save_synth_data(file_path)
        
    def _save_params(self):
        with open('data/params/heston_params.pkl', 'wb') as param_file:
            pickle.dump(self.params, param_file)

    def _load_params(self):
        with open('data/params/heston_params.pkl', 'rb') as param_file:
            self.params = pickle.load(param_file)