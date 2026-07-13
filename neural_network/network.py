import numpy as np

class FCN:
    def __init__(self, layers):
        self.layers = []
        for layer in layers:
            self.add_layer(layer)
        print("FCN initialized")

    def add_layer(self, layer):
        if len(self.layers) > 0:
            layer.initialize(self.layers[len(self.layers)-1].n)
        self.layers.append(layer)

class Layer:
    def __init__(self, neurons, activation = None):
        # Layer initialized with array of activations, neuron count and activation function name
        self.n = neurons
        self.a_vals = np.empty(neurons)

        if activation != None:
            self.activation = activation

    def initialize(self, n_prev):
        self.z_vals = np.empty(self.n)
        self.weights = np.empty((self.n, n_prev))
        self.biases = np.empty(self.n)