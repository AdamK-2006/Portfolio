from abc import ABC, abstractmethod
import numpy as np

# Abstract class

class Loss(ABC):
    @abstractmethod
    def __init__(self, delta):
        # initialisation func
        pass
    @abstractmethod
    def func(self, y_pred, y_true):
        # apply loss func to numpy array of values
        pass
    @abstractmethod
    def derive_func(self, y_pred, y_true):
        # apply loss of activation func to array of values
        pass

# Regression functions

class MSE(Loss):
    def __init__(self, delta):
        pass

    def func(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))

    def derive_func(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size

class MAE(Loss):
    def __init__(self, delta):
            pass
    
    def func(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

    def derive_func(self, y_pred, y_true):
        return np.sign(y_pred - y_true) / y_true.size

class Huber(Loss):
    def __init__(self, delta=1.0):
        self.delta = delta

    def func(self, y_pred, y_true):
        diff = y_pred - y_true
        self.is_small = np.abs(diff) <= self.delta
        squared = 0.5 * diff**2
        linear = self.delta * (np.abs(diff) - 0.5 * self.delta)
        return np.mean(np.where(self.is_small, squared, linear))

    def derive_func(self, y_pred, y_true):
        return np.where(self.is_small, (y_pred - y_true) / y_true.size, self.delta * np.sign(y_pred - y_true) / y_true.size)

# Classification functions

class CrossEntropy(Loss):
    def __init__(self, delta):
            pass
    
    def func(self, y_pred, y_true):
        return np.mean(-np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    def derive_func(self, y_pred, y_true):
        return -(y_true / (y_pred + 1e-8)) / len(y_true)