import argparse
import pandas as pd
from models.data_generating_models.gbm import GBM
from models.data_generating_models.heston import Heston
from models.data_generating_models.data_generating_model import DataGeneratingModel
from models.data_generating_models.time_gan import TimeGAN
from models.data_generating_models.quant_gan import QuantGAN
from models.data_generating_models.time_vae import TimeVAE
from config import data_generation_config, all_available_models, time_gan_config, quant_gan_config, time_vae_config

def run_model(model: DataGeneratingModel, tune: bool, generate: bool):
    if tune:
        model.fit_params_to_data()
    if generate:
        model.generate_data(save = True)


def main():
    parser = argparse.ArgumentParser(description='Tune models')
    parser.add_argument('-t', '--model-tune', nargs='+', help='List of models to tune', default=[])
    parser.add_argument('-g', '--model-generate', nargs='+', help='List of models to generate data', default=[])
    parser.add_argument('-a', '--all-tune', type=bool, help='Flag for tuning all models', default=False)
    parser.add_argument('-b', '--all-generate', type=bool, help='Flag for generating data for all models', default=False)
    
    args = parser.parse_args()

    train_data = pd.read_csv(data_generation_config["path_to_train_data"], index_col=0)
    N = data_generation_config["N"]
    M = data_generation_config["M"]

    model_map = {
        "gbm": GBM,
        "heston": Heston,
        "time_gan": TimeGAN,
        "quant_gan": QuantGAN,
        "time_vae": TimeVAE,
    }

    config_map = {
        "gbm": None,
        "heston": None,
        "time_gan": time_gan_config,
        "quant_gan": quant_gan_config,
        "time_vae": time_vae_config,
    }

    if args.all_tune:
        args.model_tune = all_available_models

    if args.all_generate:
        args.model_generate = all_available_models

    for model_str in args.model_tune:
        print(f"Tuning for model: {model_str}")
        config = config_map[model_str]
        model = model_map[model_str]
        model = model(train_data, N, M, load_params=False, config=config)
        run_model(model=model, tune=True, generate=False)
    
    for model_str in args.model_generate:
        print(f"Generating for model: {model_str}")
        config = config_map[model_str]
        model = model_map[model_str]
        model = model(train_data, N, M, load_params=True, config=config)
        run_model(model=model, tune=False, generate=True)
    
    

if __name__ == '__main__':
    # main()
    model = "heston"

    train_data = pd.read_csv("data/raw/spy_daily_closing_prices.csv", index_col=0)

    model = Heston(train_data, 30, 1000, False, None)
    model.generate_data(save=True)