import numpy as np

class FCN:
    def __init__(self, layers, loss = "MSE"):
        self.layers = []
        for layer in layers:
            self.add_layer(layer)
        self.loss = loss
        print("FCN initialized")

    def add_layer(self, layer):
        if len(self.layers) > 0:
            layer.initialize(self.layers[len(self.layers)-1].n)
        self.layers.append(layer)

    def train(self, X, y_true):
        for item in X[:5]:
            print("Running forward pass...")
            y_pred = self.forward_pass(item)
            print(y_pred)
            print("Loss: " + str(self.loss_func(y_pred, y_true)))

            print("--------------------")


    def forward_pass(self, item):
        self.layers[0].a_vals = item
        for i in range(1, len(self.layers)):
            prev_layer = self.layers[i-1]
            layer = self.layers[i]
            layer.z_vals = layer.weights @ prev_layer.a_vals + layer.biases
            layer.a_vals = layer.activation_func(layer.z_vals)
        return self.layers[-1].a_vals
    
    def mse(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))
    
    def mse_derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / len(y_true)
    
    def cross_entropy(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred))

    def loss_func(self, y_pred, y_true):
        funcs = {
        'mse': self.mse,
        'cross_entropy': self.cross_entropy
        }

        return funcs[self.loss](y_pred, y_true)
    
    def loss_func_derivative(self, y_pred, y_true):
        funcs = {
        'mse': self.mse_derivative,
        'cross_entropy': self.cross_entropy_derivative
        }

        return funcs[self.loss](y_pred, y_true)

class Layer:
    def __init__(self, neurons, activation = None):
        # Layer initialized with array of activations, neuron count and activation function name
        self.n = neurons
        self.a_vals = np.empty(neurons)

        if activation != None:
            self.activation = activation

    def initialize(self, n_prev):
        self.z_vals = np.empty(self.n)
        self.weights = np.random.randn(self.n, n_prev) * 0.01
        self.biases = np.zeros(self.n)
    
    def activation_func(self, data):
        funcs = {
        'relu': self.relu,
        'softmax': self.softmax
        }

        return funcs[self.activation](data)
    
    def relu(self, data):
        return np.maximum(0, data)
    def softmax(self, data):
        e = np.exp(data - np.max(data))
        return e / e.sum()