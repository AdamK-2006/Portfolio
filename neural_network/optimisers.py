from abc import ABC, abstractmethod
import numpy as np

# Abstract class
class Optimiser(ABC):
    @abstractmethod
    def __init__(self, learning_rate, first_moment_decay_rate, second_moment_decay_rate, epsilon):
        # initialisation func
        pass
    @abstractmethod
    def func(self, layers, batch_size):
        # apply optimiser func to network
        pass

# Implementations

class SGD(Optimiser):
    def __init__(self, learning_rate, first_moment_decay_rate, second_moment_decay_rate, epsilon):
        self.learning_rate = learning_rate

    def func(self, layers, batch_size):
        for layer in layers:
            avg_dW = layer.dW / batch_size
            avg_db = layer.db / batch_size

            layer.weights -= self.learning_rate * avg_dW
            layer.biases -= self.learning_rate * avg_db

class SGDWithMomentum(Optimiser):
    def __init__(self, learning_rate, first_moment_decay_rate, second_moment_decay_rate, epsilon):
        self.learning_rate = learning_rate
        self.first_moment_decay_rate = first_moment_decay_rate

    def func(self, layers, batch_size):
            for layer in layers:
                avg_dW = layer.dW / batch_size
                avg_db = layer.db / batch_size
                layer.vW = self.first_moment_decay_rate * layer.vW + (1-self.first_moment_decay_rate) * avg_dW
                layer.vb = self.first_moment_decay_rate * layer.vb + (1-self.first_moment_decay_rate) * avg_db

                layer.weights -= self.learning_rate * layer.vW
                layer.biases -= self.learning_rate * layer.vb

class RMSProp(Optimiser):
    def __init__(self, learning_rate, first_moment_decay_rate, second_moment_decay_rate, epsilon):
        self.learning_rate = learning_rate
        self.first_moment_decay_rate = first_moment_decay_rate
        self.second_moment_decay_rate = second_moment_decay_rate
        self.epsilon = epsilon

    def func(self, layers, batch_size):
        for layer in layers:
            avg_dW = layer.dW / batch_size
            avg_db = layer.db / batch_size
            layer.gW = self.second_moment_decay_rate * layer.gW + (1-self.second_moment_decay_rate) * (avg_dW**2)
            layer.gb = self.second_moment_decay_rate * layer.gb + (1-self.second_moment_decay_rate) * (avg_db**2)

            layer.weights -= (self.learning_rate / np.sqrt(layer.gW + self.epsilon)) * avg_dW
            layer.biases -= (self.learning_rate / np.sqrt(layer.gb + self.epsilon)) * avg_db

class Adam(Optimiser):
    def __init__(self, learning_rate, first_moment_decay_rate, second_moment_decay_rate, epsilon):
        self.learning_rate = learning_rate
        self.first_moment_decay_rate = first_moment_decay_rate
        self.second_moment_decay_rate = second_moment_decay_rate
        self.epsilon = epsilon
        self.updates = 1

    def func(self, layers, batch_size):
        for layer in layers:
            avg_dW = layer.dW / batch_size
            layer.vW = self.first_moment_decay_rate * layer.vW + (1-self.first_moment_decay_rate) * avg_dW
            v_hatW = layer.vW / (1-(self.first_moment_decay_rate**self.updates))
            layer.gW = self.second_moment_decay_rate * layer.gW + (1-self.second_moment_decay_rate) * (avg_dW**2)
            g_hatW = layer.gW / (1-(self.second_moment_decay_rate**self.updates))
            layer.weights -= (self.learning_rate * v_hatW / np.sqrt(g_hatW + self.epsilon))

            avg_db = layer.db / batch_size
            layer.vb = self.first_moment_decay_rate * layer.vb + (1-self.first_moment_decay_rate) * avg_db
            v_hatb = layer.vb / (1-(self.first_moment_decay_rate**self.updates))
            layer.gb = self.second_moment_decay_rate * layer.gb + (1-self.second_moment_decay_rate) * (avg_db**2)
            g_hatb = layer.gb / (1-(self.second_moment_decay_rate**self.updates))
            layer.biases -= (self.learning_rate * v_hatb / np.sqrt(g_hatb + self.epsilon))

        self.updates += 1