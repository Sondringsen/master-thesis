# import pandas as pd
# model = "quant_gan"

# df = pd.read_csv(f"data/processed/{model}_synth_data.csv", index_col=0)
# import matplotlib.pyplot as plt


# # Plot the first 30 timeseries
# print(df.iloc[3, :])
# for i in range(30):
#     plt.plot(df.iloc[i, :])

# plt.xlabel('Datapoints')
# plt.ylabel('Value')
# plt.title('First 30 Timeseries')
# plt.show()



# from tensorflow.python.tools import inspect_checkpoint as chkp

# chkp.print_tensors_in_checkpoint_file('data/params/timegan_model.ckpt', tensor_name='', all_tensors=False)



from src.models.data_generating_models.quant_gan import QuantGAN
import yfinance as yf
from src.config import quant_gan_config


sp500 = yf.download('^GSPC','2009-05-01','2018-12-31')


model = QuantGAN(train_data=sp500, N=30, M=1000, load_params=True, config=quant_gan_config)
# model.fit_params_to_data()
model.generate_data(save=True)

