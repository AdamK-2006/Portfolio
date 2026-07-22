import numpy as np

# fully connected network code

class FCN:
    def __init__(self, layers, loss = "mse"):
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
        for item, true in zip(X[:5], y_true[:5]):
            print("Running forward pass...")
            y_pred = self.forward_pass(item)
            print(y_pred)
            print("Loss: " + str(self.loss_func(y_pred, true)))

            self.backward_pass(true)
            print("--------------------")


    def forward_pass(self, item):
        self.layers[0].a_vals = item
        for i in range(1, len(self.layers)):
            prev_layer = self.layers[i-1]
            layer = self.layers[i]
            layer.z_vals = layer.weights @ prev_layer.a_vals + layer.biases
            layer.a_vals = layer.activation_func(layer.z_vals)
        return self.layers[-1].a_vals
    
    def backward_pass(self, y_true):
        dL_da = self.loss_func_derivative(self.layers[-1].a_vals, y_true)

        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            prev_layer = self.layers[i-1]

            dL_dz = layer.derive_z(dL_da)

            layer.dW = np.outer(dL_dz, prev_layer.a_vals)
            layer.db = dL_dz

            dL_da = layer.weights.T @ dL_dz

    # loss function code

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

# layer code

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
        self.dW = np.empty(self.n, n_prev)
        self.db = np.empty(self.n)
    
    # activation function code

    def activation_func(self, data):
        funcs = {
        'relu': self.relu,
        'softmax': self.softmax
        }

        return funcs[self.activation](data)
    
    def derive_z(self, dL_da):
        funcs = {
        'relu': self.relu_derivative,
        'softmax': self.softmax_derivative
        }

        if self.activation == 'softmax':
            return self.softmax_derivative(self.a_vals, dL_da)
        
        return dL_da * funcs[self.activation](self.a_vals)
    
    def relu(self, data):
        return np.maximum(0, data)
    #def relu_derivative(self, data):
    #    return np.maximum(0, data)
    
    def softmax(self, data):
        e = np.exp(data - np.max(data))
        return e / e.sum()
    def softmax_derivative(self, data, dL_da):
        return data * (dL_da - np.sum(dL_da * data))