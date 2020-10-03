import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample
import sklearn.linear_model as lm
from numba import jit
import sklearn.metrics as skm
import scipy as sp


@jit
def gen_def_matrix(x, y, k=1):

    xb = np.ones((x.size, 1))
    for i in range(1, k+1):
        for j in range(i+1):
            xb = np.c_[xb, (x**(i-j))*(y**j)]
    return xb

def gen_beta(data, output, lam=0):

    dim = np.shape(data)[1]
    U, sig_v, V_star = sp.linalg.svd(data)
    sig = sp.linalg.diagsvd(sig_v, U.shape[0], V_star.shape[1])
    if lam == 0:
        beta = np.conjugate(V_star).T @ sp.linalg.pinv(sig) @ np.conjugate(U).T @ output
    else:
        beta = np.linalg.inv(data.T @ data + lam * np.eye(dim)).dot(data.T.dot(output))
    
    return beta

@jit
def OLS(data_train, output_train, data_test, output_test):

    output = output_train.ravel()


    beta = gen_beta(data_train, output)

    op = data_test.dot(beta)
    output_predict = op.reshape(output_test.shape)

    MSE = mean_squared_error(output_test, output_predict)
    R2 = r2_score(output_test, output_predict)

    ot = data_train.dot(beta)

    MSEt = mean_squared_error(output, ot)
    R2t = r2_score(output, ot)
    
    return MSE, R2, MSEt, R2t, beta

@jit
def Ridge(data_train, output_train, data_test, output_test, lam=0.01):

    output = output_train.ravel()

    beta = gen_beta(data_train, output, lam)

    op = data_test.dot(beta)
    output_predict = op.reshape(output_test.shape)

    MSE = mean_squared_error(output_test, output_predict)
    R2 = r2_score(output_test, output_predict)

    ot = data_train.dot(beta)

    MSEt = mean_squared_error(output, ot)
    R2t = r2_score(output, ot)
    
    return MSE, R2, MSEt, R2t, beta

@jit
def Lasso(data_train, output_train, data_test, output_test, lam=0.0001, k=1, graph=True):

    output = output_train.ravel()

    lasso_reg = lm.Lasso(alpha=lam)

    lasso_reg.fit(data_train, output)

    op = lasso_reg.predict(data_test)
    output_predict = op.reshape(output_test.shape)

    MSE = mean_squared_error(output_test, output_predict)
    R2 = r2_score(output_test, output_predict)

    ot = lasso_reg.predict(data_train)

    MSEt = mean_squared_error(output, ot)
    R2t = r2_score(output, ot)

    
    return MSE, R2, MSEt, R2t, lasso_reg.coef_

def cross_validation(data, output, k=5, lam=0.001, method='OLS'):

    o = output.ravel()

    shf = np.random.permutation(o.size)

    d = data[shf]
    o = o[shf].reshape(output.shape)

    size_fold = int(np.ceil(len(o) / k))
    tMSE_test = 0
    tR2_test = 0
    aMSE_test = np.zeros(k)
    aR2_test = np.zeros(k)
    tMSE_train = 0
    tR2_train = 0
    aMSE_train = np.zeros(k)
    aR2_train = np.zeros(k)
    b = np.empty((k,np.shape(data)[1]))
    for i in range(k):
        start_val = size_fold*i
        end_val = min(size_fold*(i+1), len(d))
        data_test = d[start_val:end_val]
        output_test = o[start_val:end_val]
        data_train = np.r_[d[0:start_val], d[end_val:]]
        output_train = np.r_[o[0:start_val], o[end_val:]]

        if(method == 'OLS'):
            MSE, R2, MSEt, R2t, b[i] = OLS(data_train, output_train, data_test, output_test)
        elif(method == 'Ridge'):
            MSE, R2, MSEt, R2t, b[i] = Ridge(data_train, output_train, data_test, output_test, lam)
        elif(method == 'Lasso'):
            MSE, R2, MSEt, R2t, b[i] = Lasso(data_train, output_train, data_test, output_test, lam)

        tMSE_test = tMSE_test + MSE
        tR2_test = tR2_test + R2
        aMSE_test[i] = MSE
        aR2_test[i] = R2

        tMSE_train = tMSE_train + MSEt
        tR2_train = tR2_train + R2t
        aMSE_train[i] = MSEt
        aR2_train[i] = R2t

        print("The train average MSE for fold %d is: %.05f; the train average R^2-score for fold %d is: %.02f" % (i+1,MSEt,i+1, R2t))
        print("The test average MSE for fold %d is: %.05f; the test average R^2-score for fold %d is: %.02f" % (i+1,MSE,i+1, R2))

    

    preds = np.empty(k)
    bias = 0
    var = 0
    for i in range(len(data)):
        for j in range(k):
            preds[j] = b[j] @ data[i]
        ex = np.mean(preds)
        bias = bias + (output[i] - ex)**2
        var = var + np.var(preds)

    tMSE_test = tMSE_test / k
    tR2_test = tR2_test / k
    tMSE_train = tMSE_train / k
    tR2_train = tR2_train / k
    with open("results_cross_validation.txt", mode="a") as f:
        f.write("##############################################\n")
        f.write("Method used: %s; Number of folds: %d" % (method, k))
        if method != "OLS":
            f.write(";Lambda: %.06f" % lam)
        f.write("\nThe test average MSE is: %.05f; the test average R^2-score is: %.02f\n" % (tMSE_test, tR2_test))
        f.write("The train average MSE is: %.05f; the train average R^2-score is: %.02f\n" % (tMSE_train, tR2_train))
        f.write("The variance: %.05f\n" % (var/len(data)))
        f.write("The bias is: %.05f\n" % (bias/len(data)))
    print("The test average MSE is: %.05f; the test average R^2-score is: %.02f" % (tMSE_test, tR2_test))
    print("The train average MSE is: %.05f; the train average R^2-score is: %.02f" % (tMSE_train, tR2_train))
    print("The variance is: ", var/len(data))
    print("The bias is: ", bias/len(data))
    return aMSE_test, aR2_test, aMSE_train, aR2_train, tMSE_train, tR2_train, tMSE_test, tR2_test

def bootstrap(data, output, it=100, lam=0.001, method='OLS'):

    tMSE_test = 0
    tR2_test = 0
    aMSE_test = np.zeros(it)
    aR2_test = np.zeros(it)
    tMSE_train = 0
    tR2_train = 0
    aMSE_train = np.zeros(it)
    aR2_train = np.zeros(it)
    for i in range(it):
        d, o = resample(data, output)

        if(method == 'OLS'):
            MSE, R2, MSEt, R2t, b = OLS(d, o, data, output)
        elif(method == 'Ridge'):
            MSE, R2, MSEt, R2t, b = Ridge(d, o, data, output, lam)
        elif(method == 'Lasso'):
            MSE, R2, MSEt, R2t, b = Lasso(d, o, data, output, lam)
        
        tMSE_test = tMSE_test + MSE
        tR2_test = tR2_test + R2
        aMSE_test[i] = MSE
        aR2_test[i] = R2

        tMSE_train = tMSE_train + MSEt
        tR2_train = tR2_train + R2t
        aMSE_train[i] = MSEt
        aR2_train[i] = R2t
    
    tMSE_test = tMSE_test / it
    tR2_test = tR2_test / it
    tMSE_train = tMSE_train / it
    tR2_train = tR2_train / it

    print("The average train MSE is: %.05f; and the average train R2 is: %.02f" % (tMSE_train, tR2_train))
    print("The average test MSE is: %.05f; and the average test R2 is: %.02f" % (tMSE_test, tR2_test))
    return aMSE_train, aR2_train, aMSE_test, aR2_test, tMSE_train, tR2_train, tMSE_test, tR2_test
