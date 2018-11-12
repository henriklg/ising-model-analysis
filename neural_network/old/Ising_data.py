import numpy as np
import scipy.sparse as sp
np.random.seed(12)


import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

### define Ising model aprams


def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E.astype(int)



if __name__ == '__main__':
    # system size
    L=40
    n_states = 10000
    # create 10000 random Ising states
    states=np.random.choice([-1, 1], size=(n_states,L))    # (10000,40)
    # calculate Ising energies
    energies=ising_energies(states,L)
    print ("number of ising energies:",np.shape(energies))



    # reshape Ising states into RL samples: S_iS_j --> X_p
    states=np.einsum('...i,...j->...ij', states, states)
    shape=states.shape
    states=states.reshape((shape[0],shape[1]*shape[2]))
    # build final data set
    Data=[states,energies]
    print ("final data set:",len(Data[1]))


    # define number of samples
    n_samples=400
    # define train and test data sets
    X_train=Data[0][:n_samples] # (400,)
    Y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
    X_test=Data[0][n_samples:3*n_samples//2] # (200,)
    Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])

    exit(1)

'''

    from sklearn import linear_model
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import seaborn

    # set up Lasso and Ridge Regression models
    leastsq=linear_model.LinearRegression()
    ridge=linear_model.Ridge()
    lasso = linear_model.Lasso()

    # define error lists
    train_errors_leastsq = []
    test_errors_leastsq = []

    train_errors_ridge = []
    test_errors_ridge = []

    train_errors_lasso = []
    test_errors_lasso = []

    # set refularisations trength values
    lmbdas = np.logspace(-4, 5, 10)

    #Initialize coeffficients for ridge regression and Lasso
    coefs_leastsq = []
    coefs_ridge = []
    coefs_lasso=[]

    for lmbda in lmbdas:

        ### ordinary least squares
        leastsq.fit(X_train, Y_train) # fit model
        coefs_leastsq.append(leastsq.coef_) # store weights
        # use the coefficient of determination R^2 as the performance of prediction.
        train_errors_leastsq.append(leastsq.score(X_train, Y_train))
        test_errors_leastsq.append(leastsq.score(X_test,Y_test))

        ### apply Ridge regression
        ridge.set_params(alpha=lmbda) # set regularisation parameter
        ridge.fit(X_train, Y_train) # fit model
        coefs_ridge.append(ridge.coef_) # store weights
        # use the coefficient of determination R^2 as the performance of prediction.
        train_errors_ridge.append(ridge.score(X_train, Y_train))
        test_errors_ridge.append(ridge.score(X_test,Y_test))

        ### apply Ridge regression
        lasso.set_params(alpha=lmbda) # set regularisation parameter
        lasso.fit(X_train, Y_train) # fit model
        coefs_lasso.append(lasso.coef_) # store weights
        # use the coefficient of determination R^2 as the performance of prediction.
        train_errors_lasso.append(lasso.score(X_train, Y_train))
        test_errors_lasso.append(lasso.score(X_test,Y_test))

        ### plot Ising interaction J
        J_leastsq=np.array(leastsq.coef_).reshape((L,L))
        J_ridge=np.array(ridge.coef_).reshape((L,L))
        J_lasso=np.array(lasso.coef_).reshape((L,L))

        cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

        fig, axarr = plt.subplots(nrows=1, ncols=3)

        axarr[0].imshow(J_leastsq,**cmap_args)
        axarr[0].set_title('$\\mathrm{OLS}$',fontsize=16)
        axarr[0].tick_params(labelsize=16)

        axarr[1].imshow(J_ridge,**cmap_args)
        axarr[1].set_title('$\\mathrm{Ridge},\ \\lambda=%.4f$' %(lmbda),fontsize=16)
        axarr[1].tick_params(labelsize=16)

        im=axarr[2].imshow(J_lasso,**cmap_args)
        axarr[2].set_title('$\\mathrm{LASSO},\ \\lambda=%.4f$' %(lmbda),fontsize=16)
        axarr[2].tick_params(labelsize=16)

        divider = make_axes_locatable(axarr[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=fig.colorbar(im, cax=cax)

        cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
        cbar.set_label('$J_{i,j}$',labelpad=-40, y=1.12,fontsize=16,rotation=0)

        fig.subplots_adjust(right=2.0)

        plt.show()
'''
