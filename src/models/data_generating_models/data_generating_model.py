from abc import abstractmethod, ABC
import pandas as pd

class DataGeneratingModel(ABC):
    def __init__(self, train_data: pd.DataFrame, N: int, M: int, **params):
        self.train_data = train_data
        self.N = N
        self.M = M
        self.params = params
        self.synth_data = None

    @abstractmethod
    def fit_params_to_data(self):
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def generate_data(self):
        raise NotImplementedError("Subclasses should implement this method")  

    @abstractmethod
    def loss_func(self):
        raise NotImplementedError("Subclasses should implement this method")
    

    def _save_synth_data(self, path: str):
        # Should maybe have a data validation here
        self.synth_data.to_csv(path)
    
    def _save_params(self, path: str):
        self.params.to_csv(path)
    
    def _load_params(self, path: str):
        self.params = pd.read_csv(path)
