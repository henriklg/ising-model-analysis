import numpy as np
from hiddenLayer import Layer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from numba import jit

np.random.seed(13)

class MultilayerNeuralNetwork:
    def __init__(self, X_data, Y_data, n_hidden_layers, n_neurons, eta, batch_size=1):
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
        #self.output_weights = np.ones(shape=[self.n_neurons, self.n_categories])        # initial weights are all = 1
        self.output_weights = np.random.randn(self.n_neurons, self.n_categories)       # Initiate Normal Distribution
        #self.output_bias = np.zeros(shape=[1, self.n_categories]) + 0.1

        # Output
        self.a_in = None
        self.out = None

        # Training data
        self.X_train = X_data
        self.Y_train = Y_data

        self.b_error = 0
        self.batch_size = batch_size

    def create_hidden_layers(self):
        # Create the hidden layers
        self.hidden_layers = [Layer(self.n_features, self.n_neurons, self.eta)]
        for l in range(self.n_hidden_layers-1):
            self.hidden_layers.append(Layer(self.n_neurons, self.n_neurons, self.eta))

    def activation(self, a):
        # Sigmoid
        #return 1/(1 + np.exp(-z))

        # None - in regression case
        #return z

        # Softmax
        return np.exp(a)/(np.exp(a).sum(axis=1))[:, None]

    def a_derivative(self, z):
        # Linear regression
        return self.activation(z)*(1-self.activation(z))

    @jit
    def feed_forward(self):
        # Define input
        a_in = self.X_train

        # Go through all hidden layers
        for layer in self.hidden_layers:
            a_out = layer.feed_forward(a_in)
            a_in = a_out

        # Outer layer
        self.a_in = a_in
        self.z = np.matmul(self.a_in, self.output_weights) #+ self.output_bias
        self.out = self.activation(self.z)
        #print ("self.out in feed forward:",self.out)
        #print('Feed forward done!')

    def accuracy(self):
        #print ("out in accuracy:",self.out)
        Y_predict = (self.out > 0.5)
        print ("Y_predict:",Y_predict)
        a = np.sum(np.all(Y_predict == self.Y_train, axis=1))
        print(a/np.shape(Y_predict)[0])

    def accuracy_test(self, X, Y):
        a_in = X

        # Go through all hidden layers
        for layer in self.hidden_layers:
            a_out = layer.feed_forward(a_in)
            a_in = a_out

        # Outer layer
        z = np.matmul(a_in, self.output_weights) #+ self.output_bias
        out = self.activation(z)

        Y_predict = (out > 0.5)
        a = np.sum(np.all(Y_predict == Y, axis=1))
        print('Error test: ', a/np.shape(Y_predict)[0])

    def backpropagation(self):
        # Calculate outer error, number of inputs used to scale the error
        outer_error = (self.out - self.Y_train)
        print("acc:",self.accuracy())
        print("out:",self.out[4, :])
        print("Y_t:",self.Y_train[4, :])
        print("Err:",outer_error[4, :])

        # calculate errors in hidden layers
        reversed_layers = list(reversed(self.hidden_layers))
        f_weights = self.output_weights
        f_error = outer_error
        for layer in reversed_layers:
            layer.calculate_error(f_error, f_weights)
            f_weights = layer.weights
            f_error = layer.error

        # Update outer weights and bias
        self.output_weights -= self.eta * np.matmul(self.a_in.T, outer_error)
        #self.output_bias -= self.eta * np.sum(outer_error)
        for layer in reversed_layers:
            layer.backwards_propagation()
        #print ("Back propagation done!")

    def save_weights(self):
        # Save weights in order to iterate further later
        np.save('outer_weights.npy', self.output_weights)
        np.save('output_bias.npy', self.output_bias)

        counter = 1
        for layer in self.hidden_layers:
            np.save('weights_{}.npy'.format(counter), layer.weights)
            np.save('bias_{}.npy'.format(counter), layer.biases)
            counter += 1

    def test_train(self, X, Y):
        # Print MSE and R2 with regards to training data, and new input test data X, Y
        a_in = self.X_train

        for layer in self.hidden_layers:
            a_out = layer.feed_forward(a_in)
            a_in = a_out

        # Outer layer
        self.a_in = a_in
        self.out = np.matmul(self.a_in, self.output_weights) + self.output_bias
        print ("self.out in test_train:",selv.out)

        # Accuracy on training data
        MSE = np.mean((self.out - self.Y_train)**2)
        R2 = 1 - np.sum((self.Y_train - self.out)**2)/np.sum(self.Y_train-np.mean(self.Y_train)**2)
        print('MSE Training Data: ', MSE)
        print('R2 Training Data: ', R2)

        # Test data
        a_in = X

        for layer in self.hidden_layers:
            a_out = layer.feed_forward(a_in)
            a_in = a_out

        # Outer layer
        self.a_in = a_in
        self.out = np.matmul(self.a_in, self.output_weights) + self.output_bias

        # Accuracy on test data
        MSE = np.mean((self.out - Y)**2)
        R2 = 1 - np.sum((Y - self.out)**2)/np.sum((Y-np.mean(Y))**2)
        print('MSE Test Data: ', MSE)
        print('R2 Test data: ', R2)

# Functions to generate data
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
    # Load Data
    data = np.load('test_set.npy')
    print(np.shape(data))
    X = data[:, :1600]
    Y = data[:, -1].reshape(np.shape(data)[0], 1)
    Y_binary = np.c_[Y, np.where(Y==1, 0, 1)]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_binary, train_size=0.8)

    MLN = MultilayerNeuralNetwork(X_train, Y_train, n_neurons=1600, n_hidden_layers=1, eta=1e-4, batch_size=1000)

    n_epochs = 5
    for epoch in range(n_epochs):
        MLN.feed_forward()
        MLN.backpropagation()
        print ("epoch",epoch+1,"of",n_epochs)
        if (epoch%2 == 0):
            MLN.accuracy_test(X_test, Y_test)

    MLN.accuracy()

# 32 batch          # 1000
# 5 epochs          # 20
# 40 noder          # 1600
# 1 hidden layers   # 1
# 1e-2 /1e-1 eta    # 1e-6
