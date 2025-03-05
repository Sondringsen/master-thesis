import pandas as pd

df = pd.read_csv("data/processed/gbm_synth_data.csv", index_col=0)
print(df.head())
import matplotlib.pyplot as plt

# Plot the first 30 timeseries
for i in range(30):
    plt.plot(df.iloc[i, :])

plt.xlabel('Datapoints')
plt.ylabel('Value')
plt.title('First 30 Timeseries')
plt.show()