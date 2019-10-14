import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

# load data
file_name = "Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
data = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')
print('Data loaded ...')

# set spin down to -1
data[np.where(data==0)] = -1

# load labels
file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
labels = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)
print('Labels loaded ...')
# Divide into ordered, critical and disordered (X = data, Y = labels)
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]
print('Data sorted ...')

# Split data into training and test data - ignoring the critical data for now
X = np.concatenate((X_ordered, X_disordered))
Y = np.concatenate((Y_ordered, Y_disordered))
print(np.shape(X))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.20)
saveData = np.c_[X_train, Y_train]
np.save('test_set', saveData)
print('saved test set')


# Illustrate ordered vs disordered data
# L = 40
# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(X_ordered[20000].reshape(L, L), cmap='plasma_r')
# plt.title('Ordered')
#
# plt.subplot(122)
# plt.imshow(X_disordered[20000].reshape(L, L), cmap='plasma_r')
# plt.title('Disordered')
# plt.show()

# Train on the training data
logreg = LogisticRegression(X_train, Y_train.reshape(len(Y_train), 1), X_test, Y_test.reshape(len(Y_test), 1))
logreg.fit_standard()
print(logreg.accuracy())
weight = logreg.getWeights()
np.save('weights_logreg.npy', weight)
