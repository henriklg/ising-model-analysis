import tensorflow as tf
from keras import regularizers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

def to_binary(y, threshold):

    """
    Input:
        y - vector
        threshold - a number between 0 and 1.
    Return:
        A binary vector.
    """

    n = len(y)
    binary = np.zeros((n,))
    for i in range(n):
        if y[i] <= threshold:
            binary[i] = 0
        else:
            binary[i] = 1
    return binary

def read_data_set(train_size=80000,validation_size=5000):

    import pickle
    from sklearn.model_selection import train_test_split
    from keras.utils import to_categorical
    import collections

    L=40 # linear system size

    # load data
    file_name = '../Ising2DFM_reSample_L40_T=All.pkl'
    data = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
    data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
    data=data.astype('int')
    data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

    file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
    labels = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

    # divide data into ordered, critical and disordered
    X_ordered=data[:70000,:]
    Y_ordered=labels[:70000]

    X_critical=data[70000:100000,:]
    Y_critical=labels[70000:100000]

    X_disordered=data[100000:,:]
    Y_disordered=labels[100000:]

    del data,labels

    # define training and test data sets
    X=np.concatenate((X_ordered,X_disordered)) #np.concatenate((X_ordered,X_critical,X_disordered))
    Y=np.concatenate((Y_ordered,Y_disordered)) #np.concatenate((Y_ordered,Y_critical,Y_disordered))

    del X_ordered, X_disordered, Y_ordered, Y_disordered

    # pick random data points from ordered and disordered states to create the training and test sets
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_size)

    del X,Y
    return X_train,Y_train

X,y = read_data_set()
print ("X:",np.shape(X), "y:", np.shape(y))

# Parameters
Xm, Xn = X.shape
epochs = 5     # 5
eta = 0.1      # 0.1
lmbd = 0.01
n_hidden_neurons = 40
batch_size = 50 # 32

# Setting up the network
clf = tf.keras.Sequential()
clf.add(tf.keras.layers.Dense(n_hidden_neurons, activation = 'sigmoid', input_dim = Xn, kernel_regularizer = regularizers.l2(lmbd)))
clf.add(tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer = regularizers.l2(lmbd)))
sgd = tf.keras.optimizers.SGD(lr = eta)
clf.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training
history = clf.fit(X, y, epochs = epochs, batch_size = batch_size, verbose = 1)


# accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# Statistics
yhat = np.reshape(clf.predict(X), (Xm,))
yhat = to_binary(yhat, 0.5)
Accuracy_train = np.sum(y == yhat) / Xm
