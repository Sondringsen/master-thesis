import pandas as pd
import numpy as np

from scipy.optimize import fmin
from scipy.special import lambertw
from scipy.stats import kurtosis

import torch
import torch.optim as optim

from tqdm import tqdm
import random
import pickle

from models.data_generating_models.data_generating_model import DataGeneratingModel
from models.data_generating_models.quant_gan_utils import Generator, Discriminator, DatasetWrapper


class QuantGAN(DataGeneratingModel):
    def __init__(self, train_data: pd.DataFrame, N: int, M: int, load_params: bool, config: dict):
        """Initializes the Black-Scholes price paths model.
        
        Args:
            train_data (pd.DataFrame): The training data to fit the model to.
            N (int): The number of time steps.
            M (int): The number of paths to generate.
            load_params (bool): Load parameters if set to True.
            config (dict): A dictionary of config parameters.
        """
        super().__init__(train_data, N, M, load_params, config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not self.load_params:
            self.params = {
                "netG": None,
                "netD": None,
            }
        else: 
            self._load_params()

        
        random.seed(42)
        torch.manual_seed(42)

        # self.data_log = np.log(self.train_data['Close'] / self.train_data['Close'].shift(1))[1:].values
        self.data_log = np.log(self.train_data.iloc[:, 0] / self.train_data.iloc[:, 0].shift(1))[1:].values
        self.log_mean = np.mean(self.data_log)
        self.log_norm = self.data_log - self.log_mean
        self.igmm_params = self._igmm(self.log_norm)
        self.processed = self._W_delta((self.log_norm - self.igmm_params[0]) / self.igmm_params[1], self.igmm_params[2])
        self.max = np.max(np.abs(self.processed))
        self.processed /= self.max

        self.params["netG"] = Generator(self.config["nz"], 1).to(self.device)
        self.params["netD"] = Discriminator(1, 1).to(self.device)

    def _objective(self, params):
        pass

    def fit_params_to_data(self):
        self.params["netG"] = Generator(self.config["nz"], 1).to(self.device)
        self.params["netD"] = Discriminator(1, 1).to(self.device)
        
        optD = optim.RMSprop(self.params["netD"].parameters(), lr=self.config["lr"])
        optG = optim.RMSprop(self.params["netG"].parameters(), lr=self.config["lr"])

        dataset = DatasetWrapper(self.processed, self.config["seq_len"])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        t = tqdm(range(self.config["num_epochs"]))
        for epoch in t:
            for i, data in enumerate(dataloader, 0):

                self.params["netD"].zero_grad()
                real = data.to(self.device)
                batch_size, seq_len = real.size(0), real.size(1)
                noise = torch.randn(batch_size, seq_len, self.config["nz"], device=self.device)
                fake = self.params["netG"](noise).detach()

                # real = real.squeeze(-1)
                lossD = -torch.mean(self.params["netD"](real)) + torch.mean(self.params["netD"](fake))
                lossD.backward()
                optD.step()

                for p in self.params["netD"].parameters():
                    p.data.clamp_(-self.config["clip_value"], self.config["clip_value"])
        
                if i % 5 == 0:
                    self.params["netG"].zero_grad()
                    lossG = -torch.mean(self.params["netD"](self.params["netG"](noise)))
                    lossG.backward()
                    optG.step()            
            #Report metrics
            t.set_description('Loss_D: %.8f Loss_G: %.8f' % (lossD.item(), lossG.item()))
                
        # Save mode
        self._save_params()


    def generate_data(self, save = False):
        fakes = []
        for i in range(self.M):
            noise = torch.randn(1, self.N, 3, device=self.device)
            fake = self.params["netG"](noise).detach().cpu().reshape(self.N).numpy()
            fake = self._inverse(fake * self.max, self.igmm_params) + self.log_mean
            fakes.append(fake)
        fakes_df = pd.DataFrame(fakes).T.cumsum()
        self.synth_data = fakes_df.T

        if save:
            self._save_synth_data("data/processed/quant_gan_synth_data.csv")

    # def _save_params(self):
    #     torch.save(self.params["netG"], 'data/params/quantGAN/netG.pth')
    #     torch.save(self.params["netD"], 'data/params/quantGAN/netD.pth')
    #     with open('data/params/quantGAN/config.pkl', 'wb') as param_file:
    #         pickle.dump(self.config, param_file)

    # def _load_params(self):
    #     self.params = {}
    #     self.params["netG"] = torch.load('data/params/quantGAN/netG.pth', weights_only=False)
    #     self.params["netD"] = torch.load('data/params/quantGAN/netD.pth', weights_only=False)
    #     with open('data/params/quantGAN/config.pkl', 'rb') as param_file:
    #         self.config = pickle.load(param_file)

    def _save_params(self):
        torch.save(self.params["netG"].state_dict(), 'data/params/quant_gan/netG.pth')
        torch.save(self.params["netD"].state_dict(), 'data/params/quant_gan/netD.pth')
        with open('data/params/quant_gan/config.pkl', 'wb') as param_file:
            pickle.dump(self.config, param_file)

    def _load_params(self):
        self.params = {}
        self.params["netG"] = Generator(self.config["nz"], 1).to(self.device)
        self.params["netG"].load_state_dict(torch.load('data/params/quant_gan/netG.pth'))
        self.params["netD"] = Discriminator(1, 1).to(self.device)
        self.params["netD"].load_state_dict(torch.load('data/params/quant_gan/netD.pth'))
        with open('data/params/quant_gan/config.pkl', 'rb') as param_file:
            self.config = pickle.load(param_file)

    def _delta_init(self, z):
        k = kurtosis(z, fisher=False, bias=False)
        if k < 166. / 62.:
            return 0.01
        return np.clip(1. / 66 * (np.sqrt(66 * k - 162.) - 6.), 0.01, 0.48)

    def _delta_gmm(self, z):
        delta = self._delta_init(z)

        def iter(q):
            u = self._W_delta(z, np.exp(q))
            if not np.all(np.isfinite(u)):
                return 0.
            k = kurtosis(u, fisher=True, bias=False)**2
            if not np.isfinite(k) or k > 1e10:
                return 1e10
            return k

        res = fmin(iter, np.log(delta), disp=0)
        return np.around(np.exp(res[-1]), 6)
    

    def _W_delta(self, z, delta):
        return np.sign(z) * np.sqrt(np.real(lambertw(delta * z ** 2)) / delta)

    def _W_params(self, z, params):
        return params[0] + params[1] * self._W_delta((z - params[0]) / params[1], params[2])

    def _inverse(self, z, params):
        return params[0] + params[1] * (z * np.exp(z * z * (params[2] * 0.5)))

    def _igmm(self, z, eps=1e-6, max_iter=100):
        delta = self._delta_init(z)
        params = [np.median(z), np.std(z) * (1. - 2. * delta) ** 0.75, delta]
        for k in range(max_iter):
            params_old = params
            u = (z - params[0]) / params[1]
            params[2] = self._delta_gmm(u)
            x = self._W_params(z, params)
            params[0], params[1] = np.mean(x), np.std(x)

            if np.linalg.norm(np.array(params) - np.array(params_old)) < eps:
                break
            if k == max_iter - 1:
                raise "Solution not found"

        return params
            