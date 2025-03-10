# import pandas as pd
# model = "heston"

# df = pd.read_csv(f"data/processed/{model}_synth_data.csv", index_col=0)
# import matplotlib.pyplot as plt

# # Plot the first 30 timeseries
# for i in range(30):
#     plt.plot(df.iloc[i, :])

# plt.xlabel('Datapoints')
# plt.ylabel('Value')
# plt.title('First 30 Timeseries')
# plt.show()

# import pickle

# file = open(f"data/params/{model}_params.pkl", "rb")
# params = pickle.load(file)
# file.close()

# print(params)


from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file('data/params/timegan_model.ckpt', tensor_name='', all_tensors=False)

