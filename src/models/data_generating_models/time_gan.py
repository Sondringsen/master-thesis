from models.data_generating_models.data_generating_model import DataGeneratingModel
import pandas as pd
from models.data_generating_models.TimeGAN.timegan1 import train_timegan, generate_synthetic_data, train_and_generate
from models.data_generating_models.TimeGAN.data_loading import real_data_loading
import tensorflow as tf
class TimeGAN(DataGeneratingModel):
    def __init__(self, train_data: pd.DataFrame, N: int, M: int, load_params: bool, config: dict):
        """Initializes the TimeGAN model.
        
        Args:
            train_data (pd.DataFrame): The training data to fit the model to.
            N (int): The number of time steps.
            M (int): The number of paths to generate.
            load_params (bool): Load parameters if set to True.
            config (dict): A dictionary of config parameters.
        """
        transformed_train_data = real_data_loading("master", config["seq_len"])
        super().__init__(transformed_train_data, N, M, load_params, config)
        # Disable eager execution for TimeGAN
        tf.compat.v1.disable_eager_execution()

    def fit_params_to_data(self):
        train_timegan(self.train_data, self.config)

    def generate_data(self, save: bool = False):
        generated_data = train_and_generate(self.train_data, self.config, self.M)
        # generated_data = generate_synthetic_data(self.train_data, self.config, self.M)
        self.synth_data = generated_data
        if save:
            self._save_synth_data("data/processed/time_gan_synth_data.csv")

    def _objective(self) -> float:
        pass

    def _load_params(self):
        pass

    def _save_params(self):
        pass

