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
    "M": 100, # number of price paths generated
    "S0": 100, # initial price (should be loaded from the data)
    "path_to_train_data": "data/raw/spy_daily_closing_prices.csv"
}

all_available_models = ["gbm", "heston", "time_gan", "time_vae"]