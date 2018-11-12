import numpy as np
from hiddenLayer import Layer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from numba import jit

class MultilayerNeuralNetwork:
    def __init__(self, X_data, Y_data, n_hidden_layers, n_neurons, eta):
        self.X_data_full = X_data  # N x M matrix
        self.Y_data_full = Y_data  # N x 1 vector
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_neurons = n_neurons
        self.eta = eta
        self.n_categories = Y_data.shape[1]

        # Initiate hidden layer(s)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layers = []
        if n_hidden_layers > 0:
            self.create_hidden_layers()

        # Initiate outer layer
        self.output_weights = np.random.randn(self.n_neurons, self.n_categories)
        self.output_bias = np.zeros(shape=[self.n_categories, 1]) + 0.01

        # Output
        self.a_in = None
        self.out = None

    def train(self):
        iterations = 1000

        print(np.shape(self.X_data_full), np.shape(self.Y_data_full))
        X_train, X_test, Y_train, Y_test = train_test_split(self.X_data_full, self.Y_data_full, test_size=0.2)
        for i in range(iterations):
            self.feed_forward(X_train)
            self.backward_propagation(Y_train)

    def create_hidden_layers(self):
        self.hidden_layers = [Layer(self.n_features, self.n_neurons, self.eta)]
        for l in range(self.n_hidden_layers-1):
            self.hidden_layers.append(Layer(self.n_neurons, self.n_neurons, self.eta))

    def activation(self, z):
        # Sigmoid
        #return 1/(1 + np.exp(-z))

        # None
        return z

    def a_derivative(self, z):
        # Linear
        return np.ones(np.shape(z))

    @jit
    def feed_forward(self, X):
        a_in = X

        for layer in self.hidden_layers:
            a_out = layer.feed_forward(a_in)
            a_in = a_out

        # Outer layer
        self.a_in = a_in
        self.out = np.matmul(self.a_in, self.output_weights) + self.output_bias

    @jit
    def backward_propagation(self, Y):
        outer_error = (self.out - Y)/self.n_inputs

        # Update outer weights
        print('MSE ', np.sum((self.out - Y)**2)/self.n_inputs)
        # calculate errors
        reversed_layers = list(reversed(self.hidden_layers))
        f_weights = self.output_weights
        f_error = outer_error
        for layer in reversed_layers:
            layer.calculate_error(f_error, f_weights)
            f_weights = layer.weights
            f_error = layer.error

        # Update weights
        self.output_weights -= self.eta * np.matmul(self.a_in.T, outer_error)
        self.output_bias -= self.eta * np.sum(outer_error)
        for layer in reversed_layers:
            layer.backwards_propagation()


class MultilayerNeuralNetwor:
    def __init__(self, X_data, Y_data, n_hidden_layers, n_hidden_neurons, n_categories, eta):
                 #epochs, batch_size, eta, lmbds):
        self.X_data_full = X_data # N x M matrix
        self.Y_data_full = Y_data # N x 1 vector

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories
        self.n_hidden_layers = n_hidden_layers

#        self.epochs = epochs
#        self.batch_size = batch_size
#        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta

        # Create first hidden layer
        self.hidden_layers = []
        if self.n_hidden_layers > 0:
            self.create_hidden_layers()

        # Create random weights, biases
        self.create_hidden_layers()

        self.output_weights = None
        self.output_bias = None
        self.create_output_layer()

        # output/prediction
        self.a_o = None
        self.z_o = None
        self.probabilities = None

    def train(self, X_data, y_data):
        self.X_data = X_data
        self.Y_data = y_data

        self.feed_forward_training()
        self.back_propagation()

    def create_hidden_layers(self):
        self.hidden_layers = [Layer(self.n_hidden_neurons, self.n_inputs, self.eta)]
        for l in range(self.n_hidden_layers-1):
            self.hidden_layers.append(Layer(self.n_hidden_neurons, self.n_hidden_neurons, self.eta))

    def create_output_layer(self):
        self.output_weights = np.random.randn(self.n_categories, self.n_hidden_neurons)
        #print('output weights,', np.shape(self.output_weights))
        self.output_bias = np.zeros(shape=[self.n_categories, 1]) + 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid1(self, x):
        return x

    def feed_forward_training(self):
        a_h = self.X_data
        for layer in self.hidden_layers:
            out = layer.feed_forward(a_h)
            a_h = out

        self.a_o = a_h
        #print('Output weights: ', np.shape(self.output_weights))
        self.z_o = np.matmul(self.output_weights, self.a_o) + self.output_bias
        #exp_term = np.exp(self.z_o)
        self.probabilities = self.sigmoid(self.z_o)

        #self.probabilities = exp_term / np.sum(exp_term, axis=0, keepdims=True)

    def feed_forward_out(self, X):
        a_h = self.X_data
        for layer in self.hidden_layers:
            out = layer.feed_forward(a_h)
            a_h = out

        self.a_o = a_h
        self.z_o = np.matmul(self.output_weights, self.a_o) + self.output_bias
        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=0, keepdims=True)
        return self.probabilities

    def a_derivative(self, a_h):
        # Derivative of activation/sigmoid function
        return self.sigmoid(a_h)*(1-self.sigmoid(a_h))

    def back_propagation(self):
        error_output = self.probabilities - self.Y_data
        print('probs: ', self.probabilities)

        # Update output layer
        print('Error: ', error_output)
        #print('output weights: ', np.shape(self.output_weights))
        #print('a_o: ', np.shape(self.a_o))

        #print('Derivative: ', self.a_derivative(self.z_o))
        self.output_weights -= self.eta*(np.matmul(error_output*self.a_derivative(self.z_o), self.a_o.T))
        #print(self.eta*(np.matmul(error_output*self.a_derivative(self.z_o), self.a_o.T)))
        #print('Gradient outer: ', self.eta*(np.matmul(error_output*self.a_derivative(self.z_o), self.a_o.T)))
        #print(np.shape(self.eta * np.matmul(self.a_o.reshape(len(self.a_o), 1), error_output.T)))
        #self.output_weights -= np.atleast_2d(self.eta * self.a_o.T * error_output).T #* self.output_weights
        self.output_bias -= self.eta * error_output

        #print(self.output_weights)
        #self.output_weights -= self.eta*np.matmul(error_output, self.a_o.T)
        #print(self.eta*np.matmul(error_output, self.a_o.T))
        #self.output_bias -= self.eta*error_output

        reversed_layers = list(reversed(self.hidden_layers))
        f_weights = self.output_weights
        f_error = error_output
        for layer in reversed_layers:
            layer.backwards_propagation(f_error, f_weights)
            f_weights = layer.weights
            f_error = layer.error

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    # def train(self):
    #     data_indices = np.arange(self.n_inputs)
    #
    #     for i in range(self.epochs):
    #         for j in range(self.iterations):
    #             # pick datapoints with replacement
    #             chosen_datapoints = np.random.choice(
    #                 data_indices, size=self.batch_size, replace=False
    #             )
    #
    #             # minibatch training data
    #             self.X_data = self.X_data_full[chosen_datapoints]
    #             self.Y_data = self.Y_data_full[chosen_datapoints]
    #
    #             self.feed_forward()
    #             self.backpropagation()


def DesignMatrix(states):
    N = np.size(states, 0)
    size3 = (N, L ** 2)
    X = np.zeros(size3)

    for i in range(0, N):
        X[i] = np.outer(states[i, :], states[i, :]).reshape(1, -1)  # .ravel()
    return X

def ising_energies(states, L):
    J = np.zeros((L, L), )
    for i in range(L):
        J[i, (i + 1) % L] -= 1.0
    E = np.einsum('...i,ij,...j->...', states, J, states)
    # print(J.shape)
    return E

if __name__ == '__main__':
    # Generate data
    # Set random seed
    np.random.seed(62)

    # System size
    L = 40

    # Number of samples
    N = 10000
    states = np.random.choice([-1, 1], size=(N, L))
    energies = preprocessing.minmax_scale(ising_energies(states, L).reshape(-1, 1))
    X = DesignMatrix(states)
    print(np.shape(X))

    MLN = MultilayerNeuralNetwork(X, np.atleast_2d(energies), n_hidden_layers=1, n_neurons=1600, eta=1e-4) #try -7
    epochs = 10
    batch_size = 100
    iterations = N/batch_size

    data_indices = np.arange(N)
    MLN.train()
