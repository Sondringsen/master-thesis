from src.models.data_generating_models.data_generating_model import DataGeneratingModel
import pandas as pd
from src.models.data_generating_models.TimeGAN.timegan1 import train_timegan, generate_synthetic_data

class TimeGAN(DataGeneratingModel):
    def __init__(self, train_data: pd.DataFrame, N: int, M: int, **params):
        """Initializes the TimeGAN model.
        
        Args:
            train_data (pd.DataFrame): The training data to fit the model to.
            N (int): The number of time steps.
            M (int): The number of paths to generate.
            **params (dict): The parameters of the TimeGAN model.
        """
        if not params:
            params = {
                "hidden_dim": 24,
                "num_layer": 2,
                "iterations": 1000,
                "batch_size": 128,
                "module": "gru",
            }
        else:
            params = params["params"]

        super().__init__(train_data, N, M, **params)

    def fit_params_to_data(self):
        train_timegan(self.train_data, self.params)

    def generate_date(self, save: bool = False):
        generated_data = generate_synthetic_data(self.train_data, self.params)
        self.synth_data = generated_data
        if save:
            self._save_synth_data("synthetic_data.csv")

    def _objective(self) -> float:
        pass

