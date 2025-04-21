import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from models.data_generating_models.data_generating_model import DataGeneratingModel
from abc import abstractmethod, ABC
import pandas as pd
import numpy as np
from scipy.integrate import simpson
import scipy.stats as stats

class DataGeneratingModel(ABC):
    def __init__(self, train_data: pd.DataFrame, N: int, M: int, load_params: bool, config: dict = None):
        self.train_data = train_data
        try:
            self.train_data_returns = train_data.pct_change().dropna().to_numpy().flatten()
        except:
            pass
        self.N = N
        self.M = M
        self.load_params = load_params
        self.synth_data = None
        self.config = config

    @abstractmethod
    def fit_params_to_data(self):
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def generate_data(self, save: bool = False):
        raise NotImplementedError("Subclasses should implement this method")  

    @abstractmethod
    def _objective(self) -> float:
        raise NotImplementedError("Subclasses should implement this method")


    @abstractmethod
    def _save_params(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    @abstractmethod
    def _load_params(self):
        raise NotImplementedError("Subclasses should implement this method")


    def _compute_kde(self, data, grid):
        kde = stats.gaussian_kde(data)
        return kde(grid)

    def _kl_divergence(self, p, q):
        p = np.maximum(p, 1e-8)  # Avoid log(0)
        q = np.maximum(q, 1e-8)
        return simpson(p * np.log(p / q), dx=0.01)
    

    def _save_synth_data(self, path: str):
        # Should maybe have a data validation here
        self.synth_data.to_csv(path)


class TeacherForcing(DataGeneratingModel):
    def __init__(self, train_data: pd.DataFrame, N: int, M: int, load_params: bool, config: dict = None):
        super().__init__(train_data, N, M, load_params, config)
        # Create sequences of 30 days from the price data
        prices = self.train_data.iloc[:, 0].values
        self.X = np.array([prices[i:i+M] for i in range(len(prices)-M + 1)])
        self.X = self.X / self.X[:, 0].reshape(-1, 1)

    def _init_rnn(self):
        """Initialize RNN model"""
        class RNN(nn.Module):
            def __init__(self, input_size=1, hidden_size=self.config["hidden_size"], output_size=1):
                super(RNN, self).__init__()
                self.hidden_size = hidden_size
                self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
                
            def forward(self, x, hidden):
                out, hidden = self.rnn(x, hidden)
                out = self.fc(out)
                return out, hidden
        
        self.model = RNN()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        
    def _teacher_forcing_step(self, x_t, hidden, noise_scale=0.5):
        """Generate next step using RNN with teacher forcing"""
        
        # Convert input to tensor
        x_t = torch.FloatTensor([[x_t]]).unsqueeze(0)
        # Get prediction from RNN
        with torch.no_grad():
            output, hidden = self.model(x_t, hidden)
        next_val = output.squeeze().item()
        
        # Add noise
        noise = np.random.normal(0, noise_scale)
        return next_val + noise, hidden

    def generate_data(self):
        """Generate synthetic paths using RNN with teacher forcing"""
        
        if not hasattr(self, 'model'):
            self._init_rnn()
            
        # Train RNN
        X_tensor = torch.FloatTensor(self.X).unsqueeze(-1)
        for epoch in range(1000):
            self.optimizer.zero_grad()
            hidden = torch.zeros(1, len(self.X), self.model.hidden_size)
            
            # Teacher forcing: use true sequence to predict next value
            outputs, _ = self.model(X_tensor[:, :-1], hidden)
            loss = self.criterion(outputs.squeeze(), X_tensor[:, 1:].squeeze())
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss}")
            
            loss.backward()
            self.optimizer.step()
            
        # Generate synthetic paths
        synth_data = np.zeros((self.N, self.M))
        self.model.eval()
        
        for n in range(self.N):
            hidden = torch.zeros(1, 1, self.model.hidden_size)
            start_idx = np.random.randint(len(self.X))
            synth_data[n,0] = self.X[start_idx,0]
            
            for t in range(1, self.M):
                next_val, hidden = self._teacher_forcing_step(
                    synth_data[n,t-1], 
                    hidden,
                    self.config["noise_scale"]
                )
                synth_data[n,t] = next_val
        
        self.synth_data = pd.DataFrame(synth_data)

    def fit_params_to_data(self):
        pass

    def _objective(self):
        pass

    def _save_params(self):
        pass

    def _load_params(self):
        pass


if __name__ == "__main__":

    hyperparameters = {
        "noise_scale": 0.05,
        "learning_rate": 0.01,
        "hidden_size": 32
    }
    model = TeacherForcing(train_data=pd.read_csv("data/raw/spy_daily_closing_prices_train.csv", index_col=0), N=100, M=30, load_params=False, config=hyperparameters)
    model.generate_data()
    # model.synth_data[:, 0] = 0

    print(model.synth_data.shape)

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,6))
    for i in range(10):
        plt.plot(model.synth_data.loc[i,:], alpha=0.3)
    plt.title('Generated Synthetic Paths')
    plt.xlabel('Time Step')
    plt.ylabel('Value') 
    plt.grid(True)
    plt.show()


    