from abc import abstractmethod, ABC

class DataGeneratingModel(ABC):
    def __init__():
        pass

    @abstractmethod
    def generate_data(self):
        raise NotImplementedError("Subclasses should implement this method")    
    
    @abstractmethod
    def save_data(self, path):
        raise NotImplementedError("Subclasses should implement this method")