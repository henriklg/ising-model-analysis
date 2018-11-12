# import necessary packages
import numpy as np
from Ising_data import ising_energies
from NeuralNetwork import NeuralNetwork
from keras.utils import to_categorical

# variables
train_size = 0.8
test_size = 1-train_size
system_size = 40
n_states = 10000

# get test and training data
states = np.random.choice([-1, 1], size=(n_states,system_size))
energies = ising_energies(states,system_size)


# reshape Ising states into RL samples: S_iS_j --> X_p
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2])) # (10000,1600) why?

print ('ising energies:',np.shape(energies),'  ',energies[:5])
print ('ising states:',np.shape(states),'  ',states[1:,-5],'\n')

# build final data set
Data=[states,energies]


def train_test_split_numpy(inputs, labels, train_size, test_size):
    n_inputs = len(inputs)
    inputs_shuffled = inputs.copy()
    labels_shuffled = labels.copy()

    np.random.shuffle(inputs_shuffled)
    np.random.shuffle(labels_shuffled)

    train_end = int(n_inputs*train_size)
    X_train, X_test = inputs_shuffled[:train_end], inputs_shuffled[train_end:]
    Y_train, Y_test = labels_shuffled[:train_end], labels_shuffled[train_end:]

    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = train_test_split_numpy(states, energies, train_size, test_size)

"""
# define number of samples
n_samples=8000
# define train and test data setsenergies
X_train=Data[0][:n_samples] # (400,)
Y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
X_test=Data[0][n_samples:3*n_samples//2] # (200,)
Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])
"""
########################################
# one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, int(n_categories)))
    onehot_vector[range(n_inputs), integer_vector] = 1
    return onehot_vector

#Y_train_onehot, Y_test_onehot = to_categorical(Y_train), to_categorical(Y_test)
Y_train_onehot, Y_test_onehot = to_categorical(Y_train.astype(int)), to_categorical(Y_test.astype(int))


########################################
epochs = 3
batch_size = 21
eta = 2
lmbd = 0.01
n_hidden_neurons = 100
n_categories = 21

dnn = NeuralNetwork(X_train, Y_train_onehot, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)
dnn.train()
test_predict = dnn.predict(X_test)

def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)

# accuracy score from scikit library
print("Accuracy score on test set: ", accuracy_score_numpy(Y_test, test_predict))
