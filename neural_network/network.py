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

    def train(self, X, y_true, X_val = None, y_val = None, epochs=10, batch_size = 1, patience = 5):
        best_val_loss = np.inf
        epochs_without_improvement = 0

        epoch_losses = {"train": [], "val": []}
        for epoch in range(epochs):
            if epochs_without_improvement == patience:
                # load best weights
                print("Early stopping executed")
                for layer in self.layers[1:]:
                    layer.weights = layer.best_weights.copy()
                    layer.biases = layer.best_biases.copy()
                break
            iorder = np.random.permutation(len(X))
            X, y_true = X[iorder], y_true[iorder]

            total_loss = 0

            for i in range(0, len(X), batch_size):
                self.clear_gradients()
                X_batch = X[i:i+batch_size]
                y_batch = y_true[i:i+batch_size]
                for item, true in zip(X_batch, y_batch):
                    y_pred = self.predict(item)
                    total_loss += self.loss_func(y_pred, true)

                    self.backward_pass(true)

                self.sgd(len(X_batch))
            
            avg_loss = total_loss / len(X)
            epoch_losses["train"].append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            if X_val is not None and y_val is not None:
                val_loss = np.mean([self.loss_func(self.predict(x), y) 
                    for x, y in zip(X_val, y_val)])
            
                if val_loss < best_val_loss:
                    # update new best weights and biases
                    print("Model improved")
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    for layer in self.layers[1:]:
                        layer.best_weights = layer.weights.copy()
                        layer.best_biases = layer.biases.copy()
                else:
                    epochs_without_improvement += 1
                
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

            layer.dW += np.outer(dL_dz, prev_layer.a_vals)
            layer.db += dL_dz

            dL_da = layer.weights.T @ dL_dz

    def clear_gradients(self):
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
        
            layer.dW = np.zeros_like(layer.dW)
            layer.db = np.zeros_like(layer.db)
    
    # loss function code

    def mse(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))
    
    def mse_derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / len(y_true)
    
    def cross_entropy(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred + 1e-8))

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

    def sgd(self, batch_size = 1):
        for layer in self.layers[1:]:
            layer.weights -= self.learning_rate * (layer.dW / batch_size)
            layer.biases -= self.learning_rate * (layer.db / batch_size)

# layer code

class Layer:
    def __init__(self, neurons, activation = None):
        # Layer initialized with array of activations, neuron count and activation function name
        self.n = neurons
        self.a_vals = np.empty(neurons)
        self.activation = activation

    def initialize(self, n_prev):
        self.z_vals = np.empty(self.n)
        self.weights = np.random.randn(self.n, n_prev) * np.sqrt(2 / n_prev)
        self.biases = np.zeros(self.n)
        self.best_weights = np.zeros_like(self.weights)
        self.best_biases = np.zeros_like(self.biases)
        self.dW = np.zeros((self.n, n_prev))
        self.db = np.zeros(self.n)
    
    # activation function code

    def activation_func(self, data):
        if self.activation is None:
            return data

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