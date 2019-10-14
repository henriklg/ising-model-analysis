# coding: utf-8
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import scipy.linalg as scl
import scipy.sparse as sp

#for bootstrap
from sklearn.utils import resample

from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl
#from tqdm import tqdm_notebook as tqdm #import tqdm 

import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed 

from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error 
import numpy as np

def DesignMatrix(states):
    N = np.size(states, 0)
    size2 = (L,L)
    xy = np.zeros(size2)
    
    size1 = (1,L**2)
    X_design = np.zeros(size1)
    
    size3 = (N,L**2)
    X = np.zeros(size3)

    for i in range(0,N):    
        X[i] =  np.outer(states[i,:],states[i,:]).reshape(1,-1)#.ravel()
    return  X

#def get_ols_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#    u, s, v = scl.svd(x)
#    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ y

def beta_model(model, xyb, z, param = 0.5):
    #OLS
    if (model == 'Linear'): 
        betaLinear = np.linalg.pinv(xyb.T.dot(xyb)).dot(xyb.T).dot(z) 
        return betaLinear
    #Ridge
    elif (model == 'Ridge'):
        I = np.identity(np.size(xyb, 1)) 
        Lambda = param
        betaRidge = np.linalg.inv(xyb.T.dot(xyb) + Lambda*(I)).dot(xyb.T).dot(z)
        return betaRidge
    #Lasso
    elif (model == 'Lasso'):
        (M,N) = np.shape(xyb)
        X = np.c_[xyb[:,1:]]
        poly = PolynomialFeatures(degree = 1)
        X_ = poly.fit_transform(X)
        clf = linear_model.Lasso(alpha=param, max_iter=100, tol=0.001, fit_intercept=False)
        clf.fit(X_, z)
        beta = clf.coef_.reshape(N,1)
        return np.asarray(beta)
    
def bootstrap(x, y, model, param = 0.5, n_bootstrap=100):
    # Randomly shuffle data
    data_set = np.c_[y, x]
    np.random.shuffle(data_set)
    set_size = round(len(x)/5)

    # Extract test-set, never used in training. About 1/5 of total data
    x_test = data_set[0:set_size, 1:]
    y_test = data_set[0:set_size, 0]
    test_indices = np.linspace(0, set_size-1, set_size)

    # And define the training set as the rest of the data
    y_train = np.delete(data_set[:, 0], test_indices, axis = 0)
    x_train = np.delete(data_set[:, 1:], test_indices, axis = 0)

    Y_predict = []
    MSE = []
    R2s = []
    #beta = 0
    for i in range(n_bootstrap):
        x_, y_ = resample(x_train, y_train)
        beta = beta_model(model, x, y, param).reshape(1600,)
        y_hat = x_test.dot(beta)
        Y_predict.append(y_hat)
        
        # Calculate MSE and R2-score
        MSE.append(np.mean((y_test - y_hat)**2))
        R2s.append(R2(y_test, y_hat))

    # Calculate MSE, Bias and Variance
    MSE_M = np.mean(MSE)
    R2_M = np.mean(R2s)
    bias = np.mean((y_test - np.mean(Y_predict, axis=0, keepdims=True))**2)
    variance = np.mean (np.var(Y_predict, axis=0, keepdims=True))
    return MSE_M, R2_M, bias, variance
    
def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    #print(J.shape)
    return E

def predict(xyb,beta):
    """
    predicts values given a beta_model
    """
    zpredict = xyb.dot(beta)
    return zpredict 

def mu(z):
    """
    Compute mean value
    """
    n = np.size(z, 0)    
    z_mean = (1/n ) * np.sum(z)
    return z_mean
  
def calc_Variance(z, z_mu):
    """
    Compute variance
    """
    n = np.size(z, 0)    
    #Sample variance: 
    var_z = (1/n)* sum((z-z_mu)**2) 
    return var_z
        
def MSE(z, z_tilde):
    """
    compute MSE of a model
    z = real z
    z_tilde = computed z
    """
    n = np.size(z, 0)    
    #Mean Squared Error: z = true value, z_tilde = forventet z utifra modell  
    MSE = (1/n)*(sum(z-z_tilde)**2)
    #error = np.mean( np.mean((z - z_tilde)**2, axis=1, keepdims=True) )
    return MSE        
        
#def calc_R_2(y, y_tilde, y_mean):
#    n = np.size(y, 0)    
#    R_2 = 1- ((sum(y.reshape(-1,1)-y_tilde)**2)/(sum((y-y_mean)**2)))
#    return R_2

def R2(yReal, yPredicted):
    """
    compute R2-score
    """

    meanValue = np.mean(yReal)
    numerator = np.sum((yReal - yPredicted)**2)
    denominator = np.sum((yReal - meanValue)**2)
    result = 1 - (numerator/denominator)
    return result

def confidenceIntervall(z_true, z_predict, p, xyb, beta):     
    """
    Compute Confidens Intervall of a given beta-model
    """
    #N = z punkter , p = ant polynom 
    N = np.size(z_true, 0)    

    sigma2 = (1/(N-(p-1 ))) * (sum((z_true-z_predict)**2))
    # Betas varians:
    varBeta = np.linalg.pinv((xyb.T.dot(xyb)))* sigma2 
    
    # estimert standardavvik pr beta.
    betaCoeff = (np.sqrt(np.diag(varBeta))).reshape(-1,1)
    #Intervall betakoeffisienter: 
    beta_confInt = np.c_[beta-betaCoeff, beta+betaCoeff]    
    return beta, betaCoeff, beta_confInt 

L = 40 

# from https://github.com/CompPhysics/MachineLearning/blob/master/doc/Projects/2018/Project2/pdf/Project2.pdf
# Generates data: 
np.random.seed(12)
N = 10000

# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(N,L))

# calculate Ising energies
energies = ising_energies(states,L).reshape(-1,1)
#print('###    ENERGIES COMPUTED BY ISING MODEL    ####')

#find design matrix
X = DesignMatrix(states)
y = energies

#Resampler til train og test: 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#Find optimal value for lambda and alpha
y_mean = mu(y_test)
n = 100
min_value_r = 0.0001
max_value_r = 1
values_r = np.logspace(-4, 5, 10)
min_value_l = 0.000001
max_value_l = 0.001
values_l = np.logspace(-4, 5, 10)
r2_ridge = []
r2_lasso = []

for v in values_r:
    beta_ridge = beta_model('Ridge', X_train, y_train, v)
    pred_ridge = predict(X_test,beta_ridge)
    r2_ridge.append(R2(y_test, pred_ridge))

for v in values_l:
    beta_lasso = beta_model('Lasso', X_train, y_train, v)
    pred_lasso = predict(X_test,beta_lasso)
    r2_lasso.append(R2(y_test, pred_lasso))
    
plt.figure(1)
plt.title('R2 score for different alpha values for Ridge regression')    
plt.plot(np.linspace(-4,5,10), np.asarray(r2_ridge))
plt.xlabel('alpha values')
plt.ylabel('R2-score')
plt.savefig('alpha_values_Ridge')
plt.show()

plt.figure(2)
plt.title('R2 score for different lambda values for Lasso regression')    
plt.plot(np.linspace(-4,5,10), np.asarray(r2_lasso))
plt.xlabel('lambda values')
plt.ylabel('R2-score')
plt.savefig('lambda_values_Lasso')
plt.show()


#find bias and variance

MSE_M_OLS, R2_M_OLS, bias_OLS, variance_OLS =  bootstrap(X_train, y_train,'Linear', 1)
MSE_M_R, R2_M_R, bias_R, variance_R =  bootstrap(X_train, y_train,'Ridge', 0.5)
MSE_M_L, R2_M_L, bias_L, variance_L =  bootstrap(X_train, y_train,'Lasso', 0.002,)

print('Ordinary Least squares')
print('MSE: ', MSE_M_OLS)
print('R2: ', R2_M_OLS)
print('bias: ',bias_OLS)
print('variance: ', variance_OLS)

print('Ridge Regression')
print('MSE: ', MSE_M_R)
print('R2: ', R2_M_R)
print('bias: ',bias_R)
print('variance: ', variance_R)

print('Lasso Regression')
print('MSE: ', MSE_M_L)
print('R2: ', R2_M_L)
print('bias: ',bias_L)
print('variance: ', variance_L)
"""


"""
# Finner beta for train vha. OLS: 
beta_OLS = beta_model('Linear', X_train, y_train)
beta_Ridge = beta_model('Ridge', X_train, y_train, 0.5)
beta_Lasso = beta_model('Lasso', X_train, y_train, 0.002)


#trener modellen: 
predicted_energies_OLS = predict(X_test,beta_OLS)
predicted_energies_Ridge = predict(X_test,beta_Ridge)
predicted_energies_Lasso = predict(X_test,beta_Lasso)

# Finn MSE: 
mse_OLS = MSE(y_test, predicted_energies_OLS)
mse_Ridge = MSE(y_test, predicted_energies_Ridge)
mse_Lasso = MSE(y_test, predicted_energies_Lasso)

print('###   MSE OLS   ###')
print(mse_OLS)
print('###   MSE Ridge   ###')
print(mse_Ridge)
print('###   MSE Lasso   ###')
print(mse_Lasso)

y_mean = mu(y_test)
print('###   my_y   ###')
print(y_mean)

R2_OLS = R2(y_test, predicted_energies_OLS)
R2_Ridge = R2(y_test, predicted_energies_Ridge)
R2_Lasso = R2(y_test, predicted_energies_Lasso)

print('###   R2 OLS   ###')
print(R2_OLS)
print('###   R2 Ridge   ###')
print(R2_Ridge)
print('###   R2 Lasso   ###')
print(R2_Lasso)


beta, betaCoeff, beta_confInt = confidenceIntervall(y_test, energiesPredicted, 1, X_test, beta)

print(betaCoeff.shape) 
print(beta.shape)
print(beta_confInt) 