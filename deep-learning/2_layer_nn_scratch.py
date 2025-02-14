import numpy as np

# Define activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# Neural Network Class
class TwoLayerNN:
    def __init__(self, n_x, n_h, n_y, activation_func=sigmoid):
        """
        Initializes a 2-layer Neural Network.
        :param n_x: Number of input features
        :param n_h: Number of neurons in hidden layer
        :param n_y: Number of output neurons (usually 1 for binary classification)
        :param activation_func: Activation function for hidden layer (default: sigmoid)
        """
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.activation_func = activation_func

        # Initialize weights and biases
        self.parameters = {
            "W1": np.random.randn(n_h, n_x) * 0.01,  # (n_h, n_x)
            "b1": np.zeros((n_h, 1)),  # (n_h, 1)
            "W2": np.random.randn(n_y, n_h) * 0.01,  # (n_y, n_h)
            "b2": np.zeros((n_y, 1)),  # (n_y, 1)
        }

    def forward_propagation(self, X):
        """
        Performs forward propagation through the network.
        :param X: Input data of shape (n_x, m) where m is number of examples
        :return: Output A2 and a cache of all intermediate calculations
        """
        W1, b1, W2, b2 = self.parameters["W1"], self.parameters["b1"], self.parameters["W2"], self.parameters["b2"]

        # Hidden Layer
        Z1 = np.dot(W1, X) + b1  # (n_h, m)
        A1 = self.activation_func(Z1)  # Apply activation (default: sigmoid)

        # Output Layer
        Z2 = np.dot(W2, A1) + b2  # (n_y, m)
        A2 = sigmoid(Z2)  # Sigmoid for output probability

        # Store cache for backpropagation
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

        return A2, cache

# Example Usage
np.random.seed(42)
X_example = np.random.randn(3, 5)  # 3 input features, 5 training examples
nn = TwoLayerNN(n_x=3, n_h=4, n_y=1)  # 3 input neurons, 4 hidden neurons, 1 output neuron
output, cache = nn.forward_propagation(X_example)

print("Output of Neural Network:\n", output)
