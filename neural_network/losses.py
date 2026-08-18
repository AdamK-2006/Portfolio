from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    @abstractmethod
    def func(self, y_pred, y_true):
        # apply loss func to numpy array of values
        pass
    @abstractmethod
    def derive_func(self, y_pred, y_true):
        # apply loss of activation func to array of values
        pass

class MSE(Loss):
    def func(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))

    def derive_func(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size

class Cross_Entropy(Loss):
    def func(self, y_pred, y_true):
        return np.mean(-np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    def derive_func(self, y_pred, y_true):
        return -(y_true / (y_pred + 1e-8)) / len(y_true)