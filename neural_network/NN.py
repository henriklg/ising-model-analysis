import numpy as np
#from hiddenLayer import Layer
from sklearn import preprocessing
from NN import MultilayerNetwork
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error

def ising_energies(states,L):
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    E = np.einsum('...i,ij,...j->...',states,J,states)
    #print(J.shape)
    return E

def DesignMatrix(states):
    N = np.size(states, 0)
    size3 = (N,L**2)
    X = np.zeros(size3)

    for i in range(0,N):
        X[i] =  np.outer(states[i,:],states[i,:]).reshape(1,-1)#.ravel()
    return  X


if __name__ == '__main__':
    np.random.seed(100)

    # System size
    L = 40

    # Number of samples
    N = 10000
    states = np.random.choice([-1, 1], size=(N, L))
    energies = preprocessing.minmax_scale(ising_energies(states, L).reshape(-1, 1))
    X = DesignMatrix(states)

    X_train, X_test, Y_train, Y_test = train_test_split(X, energies, test_size=0.2)

    network1 = MultilayerNetwork(X_train, np.atleast_2d(Y_train), n_hidden_layers=3, n_neurons=1600, eta=1e-6)
    network1.train()
