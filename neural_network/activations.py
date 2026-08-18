from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    @abstractmethod
    def func(self, data):
        # apply activation func to numpy array of values
        pass
    @abstractmethod
    def derive_func(self, data):
        # apply derivation of activation func to array of values
        pass

class ReLu(Activation):
    def func(self, data):
        return np.maximum(0, data)
    def derive_func(self, data):
        return (data > 0).astype(float)

class Softmax(Activation):
    def func(self, data):
        e = np.exp(data - np.max(data, axis=-1, keepdims=True))
        self.output = e / e.sum(axis=-1, keepdims=True)
        return self.output
    
    def derive_func(self, dL_da):
        return self.output * (dL_da - np.sum(dL_da * self.output, axis=-1, keepdims=True))