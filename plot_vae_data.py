import numpy as np
import matplotlib.pyplot as plt

def load_data(data_dir: str, dataset: str) -> np.ndarray:
    """
    Load data from a dataset located in a directory.

    Args:
        data_dir (str): The directory where the dataset is located.
        dataset (str): The name of the dataset file (without the .npz extension).

    Returns:
        np.ndarray: The loaded dataset.
    """
    return get_npz_data("src/models/data_generating_models/TimeVAE/data/stockv_subsampled_train_perc_100.npz")

def get_npz_data(input_file: str) -> np.ndarray:
    """
    Load data from a .npz file.

    Args:
        input_file (str): The path to the .npz file.

    Returns:
        np.ndarray: The data array extracted from the .npz file.
    """
    loaded = np.load(input_file)
    return loaded["data"]

data = load_data("", "")
print(data.shape)
# Plot num_datapoints over time for the first feature
# num_datapoints = data.shape[0]

# # Create a time array
# time = np.arange(num_datapoints)

# # Plot the first feature over time
# fig, axs = plt.subplots(6, 1, figsize=(10, 8))

# for i in range(6):
#     axs[i].plot(time, data[:, 0, i])
#     axs[i].set_xlabel('Time')
#     axs[i].set_ylabel(f'Feature {i+1} Value')
#     axs[i].set_title(f'Feature {i+1} Over Time')

# plt.tight_layout()

# plt.xlabel('Time')
# plt.ylabel('First Feature Value')
# plt.title('First Feature Over Time')
# plt.show()