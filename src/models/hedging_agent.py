from abc import ABC, abstractmethod

class HedgingAgent(ABC):
    def __init__(self, option, data, K, r):
        self.option = option
        self.data = data
        self.K = K
        self.r = r
        self.payoff = option.calculate_payoff(data)

    @abstractmethod
    def loss(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    @abstractmethod
    def get_action(self):
        raise NotImplementedError("Subclasses should implement this method")