import numpy as np
from activations import *
from losses import *
from optimisers import *

# fully connected network code

class FCN:
    def __init__(self, layers, optimiser = "sgd", learning_rate = 0.01, loss = "mse", first_moment_decay_rate = 0.9, second_moment_decay_rate = 0.999, epsilon = 1e-8):
        self.layers = []
        for layer in layers:
            self.add_layer(layer)

        if loss is None:
            self.loss = None
        else:
            funcs = {
                'mse': MSE,
                'cross_entropy': Cross_Entropy
            }
            self.loss = funcs[loss]()

        if optimiser is None:
            self.optimiser = None
        else:
            optimisers = {
                'sgd': SGD,
                'sgd_with_momentum': SGD_With_Momentum,
                'rmsprop': RMSProp,
                'adam': Adam
            }
            self.optimiser = optimisers[optimiser](learning_rate, first_moment_decay_rate, second_moment_decay_rate, epsilon)

        print("FCN initialized")

    def add_layer(self, layer):
        if len(self.layers) > 0:
            layer.initialize(self.layers[-1].n)
        self.layers.append(layer)

    def train(self, X, y_true, X_val = None, y_val = None, epochs=10, batch_size = 1, patience = 5):
        best_val_loss = np.inf
        epochs_without_improvement = 0

        epoch_losses = {"train": [], "val": [], "val_accuracy": []}
        for epoch in range(epochs):
            iorder = np.random.permutation(len(X))
            X, y_true = X[iorder], y_true[iorder]

            total_loss = 0

            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y_true[i:i+batch_size]
                y_pred = self.predict(X_batch)
                total_loss += self.loss.func(y_pred, y_batch)

                self.backward_pass(y_batch)

                self.optimiser.func(self.layers[1:], len(X_batch))
            
            avg_loss = total_loss / len(X)
            epoch_losses["train"].append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self.loss.func(val_pred, y_val)
            
                if val_loss < best_val_loss:
                    # update new best weights and biases
                    print("Model improved!")
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    for layer in self.layers[1:]:
                        layer.best_weights = layer.weights.copy()
                        layer.best_biases = layer.biases.copy()
                else:
                    epochs_without_improvement += 1
                
                epoch_losses["val"].append(val_loss)

                val_preds = np.argmax(val_pred, axis=1)
                val_true = np.argmax(y_val, axis=1)
                val_acc = np.mean(val_preds == val_true) * 100
                epoch_losses["val_accuracy"].append(val_acc)

                print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.2f}%")

                if epochs_without_improvement == patience:
                    # load best weights
                    print("Early stopping executed")
                    for layer in self.layers[1:]:
                        layer.weights = layer.best_weights.copy()
                        layer.biases = layer.best_biases.copy()
                    break
            print("...")
        return epoch_losses

    def predict(self, X):
        self.layers[0].a_vals = X
        for i in range(1, len(self.layers)):
            prev_layer = self.layers[i-1]
            layer = self.layers[i]
            layer.z_vals = prev_layer.a_vals @ layer.weights.T + layer.biases
            layer.a_vals = layer.activation.func(layer.z_vals)
        return self.layers[-1].a_vals

    def backward_pass(self, y_true):
        dL_da = self.loss.derive_func(self.layers[-1].a_vals, y_true)
        # this is of shape (batch_size, neurons)

        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            prev_layer = self.layers[i-1]

            dL_dz = layer.derive_z(dL_da)
            layer.dW = dL_dz.T @ prev_layer.a_vals
            layer.db = np.sum(dL_dz, axis=0)

            dL_da = dL_dz @ layer.weights

# layer code

class Layer:
    def __init__(self, neurons, activation = None):
        # Layer initialized with array of activations, neuron count and activation function name
        self.n = neurons
        self.a_vals = np.empty(neurons)
        if activation is None:
            self.activation = None
        else:
            funcs = {
                'relu': ReLu, 'softmax': Softmax
            }
            self.activation = funcs[activation]()

    def initialize(self, n_prev):
        self.z_vals = np.empty(self.n)
        self.weights = np.random.randn(self.n, n_prev) * np.sqrt(2 / n_prev)
        self.biases = np.zeros(self.n)
        self.best_weights = np.zeros_like(self.weights)
        self.best_biases = np.zeros_like(self.biases)
        self.dW = np.zeros((self.n, n_prev))
        self.db = np.zeros(self.n)

        self.vW = np.zeros((self.n, n_prev))  # momentum velocity
        self.vb = np.zeros(self.n)

        self.gW = np.zeros((self.n, n_prev))  # rmsprop cache
        self.gb = np.zeros(self.n)
    
    # activation function code
    
    def derive_z(self, dL_da):
        if self.activation is None:
            return dL_da
        if isinstance(self.activation, Softmax):
            return self.activation.derive_func(dL_da)
        return dL_da * self.activation.derive_func(self.z_vals)