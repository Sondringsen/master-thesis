from models.data_generating_models.data_generating_model import DataGeneratingModel
from models.data_generating_models.TimeVAE.src.vae.timevae import TimeVAE
import pandas as pd
import numpy as np
import os


from models.data_generating_models.TimeVAE.src.data_utils import (
    load_yaml_file,
    scale_data,
    inverse_transform_data,
    save_scaler,
    load_scaler,
    save_data,
)
import models.data_generating_models.TimeVAE.src.paths as paths
from models.data_generating_models.TimeVAE.src.vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    save_vae_model,
    get_posterior_samples,
    get_prior_samples,
    load_vae_model,
)

class TimeVAE(DataGeneratingModel):
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
        self.reshape_to_sequences()

        if not self.load_params:
            self.params = {
                "model": None,
                "scaler": None,
            }
            self.scaled_train_data, _, self.params["scaler"] = scale_data(self.train_data)
            self.hyperparameters = load_yaml_file(paths.HYPERPARAMETERS_FILE_PATH)["timeVAE"]
            _, sequence_length, feature_dim = self.scaled_train_data.shape
            self.params["model"] = instantiate_vae_model(
                vae_type="timeVAE",
                sequence_length=sequence_length,
                feature_dim=feature_dim,
                **self.hyperparameters,
            )
        else:
            self._load_params()
        
        

    def fit_params_to_data(self):
        train_vae(
            vae=self.params["model"],
            train_data=self.scaled_train_data,
            max_epochs=10,
            verbose=1,
        )
        self._save_params()
    

    def generate_data(self, save = False):
        prior_samples = get_prior_samples(self.params["model"], num_samples=self.M)
        inverse_scaled_prior_samples = inverse_transform_data(prior_samples, self.params["scaler"])
        synth_data = np.squeeze(inverse_scaled_prior_samples, -1)
        self.synth_data = pd.DataFrame(synth_data)
        if save:
            self._save_synth_data("data/processed/time_vae_synth_data.csv")


    def _save_params(self):
        save_scaler(scaler=self.params["scaler"], dir_path="data/params/time_vae")
        save_vae_model(vae=self.params["model"], dir_path="data/params/time_vae")

    def _load_params(self):
        self.params = {}
        loaded_model = load_vae_model("timeVAE", "data/params/time_vae")
        self.params["model"] = loaded_model
        scaler = load_scaler("data/params/time_vae")
        self.params["scaler"] = scaler

    def reshape_to_sequences(self):
        """
        Reshape data from (num_samples, num_features) to (n_samples, n_timesteps, n_features)
        
        Parameters:
            data (numpy.ndarray): Input data of shape (num_samples, num_features)
            n_timesteps (int): Sequence length
        
        Returns:
            numpy.ndarray: Reshaped data of shape (n_samples, n_timesteps, n_features)
        """
        data = self.train_data.copy()
        num_samples, num_features = data.shape
        n_samples = num_samples - self.config["seq_len"] + 1
        
        if n_samples <= 0:
            raise ValueError("n_timesteps must be smaller than or equal to num_samples")
        
        reshaped_data = np.array([
            data[i:i + self.config["seq_len"]] for i in range(n_samples)
        ])
        
        self.train_data = reshaped_data

    def _objective(self):
        pass