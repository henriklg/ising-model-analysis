import numpy as np

class Layer():

    """
    Hidden layer l for a multilayer perception model
    """

    def __init__(self, n_neurons, n_features, eta):
        self.n_neurons = n_neurons
        self.n_features = n_features

        self.weights = np.random.randn(self.n_features, self.n_neurons)
        self.biases = np.zeros(self.n_neurons) + 0.01
        self.eta = eta

        self.z = None
        self.a_out = None
        self.a_in = None

    def sigmoid(self, u):
        return 1/(1+np.exp(-u))

    def a_derivative(self, a_h):
        # Derivative of activation/sigmoid function
        return a_h*(1-a_h)

    def feed_forward(self, a_in):
        # feed-forward for output
        self.a_in = a_in
        self.z = np.matmul(self.a_in, self.weights) + self.bias
        self.a_out = self.sigmoid(self.z)

        return self.a_h

    def backwards_propagation(self, f_error, f_weights):
        # f_error, f_weights from layer l+1

        # Calculate error
        error_hidden = np.matmul(f_error, f_weights.T)*self.a_derivative(self.a_h)

        # update weights and bias
        self.weights -= self.eta*np.matmul(self.a_in.T, error_hidden)
        self.biases -= self.eta*error_hidden
