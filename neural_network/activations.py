from abc import ABC, abstractmethod
import numpy as np

# Abstract class

class Activation(ABC):
    @abstractmethod
    def func(self, data):
        # apply activation func to numpy array of values
        pass
    @abstractmethod
    def derive_func(self, data):
        # apply derivation of activation func to array of values
        pass

# Hidden layer activations

class ReLu(Activation):
    def func(self, data):
        return np.maximum(0, data)
    def derive_func(self, data):
        return (data > 0).astype(float)

class LeakyReLu(Activation):
    def func(self, data):
        return np.maximum(0.01 * data, data)
    def derive_func(self, data):
        return np.where(data > 0, 1.0, 0.01)

class Sigmoid(Activation):
    def func(self, data):
        e = np.exp(-data)
        self.output = 1 / (1 + e)
        return self.output
    def derive_func(self, data):
        e = np.exp(-data)
        return self.output * (1-self.output)

class Tanh(Activation):
    def func(self, data):
        e_px = np.exp(data)
        e_nx = np.exp(-data)
        self.output = (e_px - e_nx) / (e_px + e_nx)
        return self.output
    def derive_func(self, data):
        return 1 - (self.output**2)

# Output layer activations

class Softmax(Activation):
    def func(self, data):
        e = np.exp(data - np.max(data, axis=-1, keepdims=True))
        self.output = e / e.sum(axis=-1, keepdims=True)
        return self.output
    
    def derive_func(self, dL_da):
        return self.output * (dL_da - np.sum(dL_da * self.output, axis=-1, keepdims=True))