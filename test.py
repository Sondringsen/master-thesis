import pandas as pd
import numpy as np


data = pd.read_csv("data/raw/spy_daily_closing_prices_train.csv", index_col=0)
data 
data = np.array(data.values)
np.save("data/raw/spy_daily_closing_prices_train_values.npy", data)

# data = np.save("src/models/data_generating_models/Diffusion-TS-main/master_thesis/samples/master_thesis_ground_truth_30_train.npy")
# print(data.shape)

