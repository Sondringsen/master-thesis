option_config = {
    "S": 100, # initial price
    "K": 100,  # strike price
    "mu": 0.1, # drift
    "sigma": 0.2, # volatility of the Brownian motion
    "T": 1/12, #  time to maturity
    "r": 0.05 # rate of return
}

monte_carlo_config = {
    "N": 1000, # number of time steps
    "M": 10, # number of price paths generated
}

hedging_agent_config = {

}