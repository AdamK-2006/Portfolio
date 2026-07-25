import numpy as np

# fully connected network code

class FCN:
    def __init__(self, layers, optimizer = "sgd", learning_rate = 0.01, loss = "mse"):
        self.layers = []
        for layer in layers:
            self.add_layer(layer)
        self.loss = loss
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        print("FCN initialized")

    def add_layer(self, layer):
        if len(self.layers) > 0:
            layer.initialize(self.layers[len(self.layers)-1].n)
        self.layers.append(layer)

    def train(self, X, y_true, X_val = None, y_val = None, epochs=10):
        epoch_losses = {"train": [], "val": []}
        for epoch in range(epochs):
            total_loss = 0
            for item, true in zip(X, y_true):
                y_pred = self.predict(item)
                total_loss += self.loss_func(y_pred, true)

                self.backward_pass(true)

                self.sgd()
            
            avg_loss = total_loss / len(X)
            epoch_losses["train"].append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            if X_val is not None and y_val is not None:
                val_loss = np.mean([self.loss_func(self.predict(x), y) 
                    for x, y in zip(X_val, y_val)])
                epoch_losses["val"].append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}")
        return epoch_losses

    def predict(self, item):
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

    # TO DO!!!
    def cross_entropy_derivative(self, y_pred, y_true):
        return -(y_true / (y_pred + 1e-8))

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

    # optimizer code

    def sgd(self):
        for layer in self.layers[1:]:
            layer.weights = layer.weights - (self.learning_rate * layer.dW)
            layer.biases = layer.biases - (self.learning_rate * layer.db)

# layer code

class Layer:
    def __init__(self, neurons, activation = None):
        # Layer initialized with array of activations, neuron count and activation function name
        self.n = neurons
        self.a_vals = np.empty(neurons)
        self.activation = activation

    def initialize(self, n_prev):
        self.z_vals = np.empty(self.n)
        self.weights = np.random.randn(self.n, n_prev) * 0.01
        self.biases = np.zeros(self.n)
        self.dW = np.empty((self.n, n_prev))
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
        
        return dL_da * funcs[self.activation](self.z_vals)
    
    def relu(self, data):
        return np.maximum(0, data)
    def relu_derivative(self, data):
        return (data > 0).astype(float)
    
    def softmax(self, data):
        e = np.exp(data - np.max(data))
        return e / e.sum()
    def softmax_derivative(self, data, dL_da):
        return data * (dL_da - np.sum(dL_da * data))