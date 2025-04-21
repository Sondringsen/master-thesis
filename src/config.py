option_config = {
    "S": 100, # initial price
    "K": 100,  # strike price
    "mu": 0.1, # drift
    "sigma": 0.2, # volatility of the Brownian motion
    "T": 1/12, #  time to maturity
    "r": 0.05 # rate of return
}


hedging_agent_config = {

}

data_generation_config = {
    "N": 30, # number of time steps
    "M": 100_000, # number of price paths generated
    "path_to_train_data": "data/raw/spy_daily_closing_prices.csv"
}

time_gan_config = {
    "data_name": "master",
    "module": "gru",
    "hidden_dim": 24,
    "num_layer": 3,
    "iterations": 10000,
    "batch_size": 128,
    "seq_len": 24,
}

quant_gan_config = {
    "clip_value": 0.01,
    "lr": 0.0002,
    "num_epochs": 10,
    "nz": 3,
    "batch_size": 30,
    "seq_len": 127,
}

time_vae_config = {
    "sequence_length": 30,
    "latent_dim": 8,
    "hidden_layer_sizes": [8, 8, 8],
    "reconstruction_wt": 3.0,
    "batch_size": 16,
    "use_residual_conn": True,
    "trend_poly": 0,
    "custom_seas": None
}

all_available_models = ["gbm", "heston", "time_gan", "time_vae"]