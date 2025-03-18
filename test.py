import pandas as pd
import pickle
import matplotlib.pyplot as plt


model = "heston"

df = pd.read_csv(f"data/processed/{model}_synth_data.csv", index_col=0)
# df = 400*(1+df.cumsum(axis=1))

# Plot the first 30 timeseries
# print(df.iloc[3, :])
for i in range(30):
    plt.plot(df.iloc[i, :])

plt.xlabel('Datapoints')
plt.ylabel('Value')
plt.title('First 30 Timeseries')
plt.show()


with open("data/params/heston_params.pkl", "rb") as file:
    heston_params = pickle.load(file)

print(heston_params)