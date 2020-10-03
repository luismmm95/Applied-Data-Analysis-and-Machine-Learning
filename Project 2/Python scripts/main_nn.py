from __future__ import absolute_import, division, print_function
import numpy as np
seed=12
np.random.seed(seed)
import numpy as np
import mlp

import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

np.random.seed() # shuffle random seed generator

# Ising model parameters
L=40 # linear system size
J=-1.0 # Ising interaction
T=np.linspace(0.25,4.0,16) # set of temperatures
T_c=2.26 # Onsager critical temperature in the TD limit

##### prepare training and test data sets

import pickle,os
from sklearn.model_selection import train_test_split

###### define ML parameters
num_classes=2
train_to_test_ratio=0.5 # training samples

# load data
file_name = "Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
data = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')
data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
labels = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]
print(Y_ordered)

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]
print(Y_critical)

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]
print(Y_disordered)

del data,labels

# define training and test data sets
X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))

# pick random data points from ordered and disordered states 
# to create the training and test sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio)

# full data set
X=np.concatenate((X_critical,X))
Y=np.concatenate((Y_critical,Y))

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print()
print(X_train.shape[0], 'train samples')
print(X_critical.shape[0], 'critical samples')
print(X_test.shape[0], 'test samples')

##### plot a few Ising states
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# set colourbar map
cmap_args=dict(cmap='plasma_r')

# plot states
fig, axarr = plt.subplots(nrows=1, ncols=3)

axarr[0].imshow(X_ordered[20001].reshape(L,L),**cmap_args)
axarr[0].set_title('$\\mathrm{ordered\\ phase}$',fontsize=16)
axarr[0].tick_params(labelsize=16)

axarr[1].imshow(X_critical[10001].reshape(L,L),**cmap_args)
axarr[1].set_title('$\\mathrm{critical\\ region}$',fontsize=16)
axarr[1].tick_params(labelsize=16)

im=axarr[2].imshow(X_disordered[50001].reshape(L,L),**cmap_args)
axarr[2].set_title('$\\mathrm{disordered\\ phase}$',fontsize=16)
axarr[2].tick_params(labelsize=16)

fig.subplots_adjust(right=2.0)

plt.show()


#### Creating the one-hot encoded target vectors
X_cross = X_train

temp = np.zeros((len(Y_train), 2))
order = np.zeros(2)
order[0] = 1
disorder = np.zeros(2)
disorder[1] = 1
for i in range(len(Y_train)):
    if Y_train[i] == 1:
        temp[i] = order
    if Y_train[i] == 0:
        temp[i] = disorder

Y_train = temp
temp = np.zeros((len(Y_test), 2))
for i in range(len(Y_test)):
    if Y_test[i] == 1:
        temp[i] = order
    if Y_test[i] == 0:
        temp[i] = disorder
Y_test = temp

Y_cri = np.empty((len(Y_critical),2))
for i in range(len(Y_cri)):
    if Y_critical[i] == 1:
        Y_cri[i] = order
    else:
        Y_cri[i] = disorder

Y_cross = Y_train

#### Creating the training and validation data

val_size = 16250

X_valid = X_train[-val_size:]
Y_valid = Y_train[-val_size:]

X_train = X_train[:-val_size]
Y_train = Y_train[:-val_size]

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'validation samples')
print(X_critical.shape[0], 'critical samples')
print(X_test.shape[0], 'test samples')

#### Cross-validation for the Neural Networks

def cross_validation(data, targets, test, test_targets, nhidden=5, k=10):

    data_sp = np.split(data, k)
    size = int(len(data)/k)
    targets_sp = np.split(targets, k)
    errs = np.empty(k)
    accs = np.empty(k)
    
    for i in range(k):
        t_ind = np.arange(len(data))
        t_ind = np.delete(t_ind, t_ind[i*size:(i+1)*size])
        print(t_ind)
        valid = data_sp[i]
        valid_targets = targets_sp[i]
        train = data[t_ind]
        train_targets = targets[t_ind]
        print(train.shape)
        print(train_targets.shape)

        net = mlp.mlp(train, train_targets, nhidden)

        val_err, val_acc, tr_err, tr_acc = net.earlystopping(train, train_targets, valid, valid_targets, 50)

        plt.plot(val_err)
        plt.figure()
        plt.plot(val_acc)
        plt.figure()
        plt.plot(tr_err)
        plt.figure()
        plt.plot(tr_acc)
        plt.show()

        errs[i], accs[i] = net.test(test, test_targets)

        print("Error of %d fold is: " % (i+1), errs[i])
        print("Accuracy of %d fold is: " % (i+1), accs[i])

    print("Mean error is: ", np.mean(errs))
    print("Standard deviation of error is: ", np.std(errs))
    print("Mean accuracy is: ", np.mean(accs))
    print("Standard deviation of accuracy is: ", np.std(accs))

#### Running the Neural Network and plotting its learning

net = mlp.mlp(X_train, Y_train, 5)

val_err, val_acc, tr_err, tr_acc = net.earlystopping(X_train, Y_train, X_valid, Y_valid, 100)

plt.plot(val_err)
plt.plot(tr_err)
plt.figure()
plt.plot(val_acc)
plt.plot(tr_acc)
print("################################")
print("Test Results")
net.test(X_test, Y_test)
print("################################")
print("Critical Results")
net.test(X_critical, Y_cri)
plt.show()

#### Cross-validation

cross_validation(X_cross, Y_cross, X_test, Y_test, nhidden=4)

#### Testing different parameters for hidden nodes and learning rates

def plot_data(x,y,data):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)
    fig.colorbar(cax)

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format( 100*data[j,i])  
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)

    ax.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)

    plt.tight_layout()

    plt.show()

etas = [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 0.1]
nodes = [1, 5, 10, 20]

ltr_acc = np.empty((len(nodes),len(etas)))
ltr_err = np.empty((len(nodes),len(etas)))
test_acc = np.empty((len(nodes),len(etas)))
test_err = np.empty((len(nodes),len(etas)))
cri_acc = np.empty((len(nodes),len(etas)))
cri_err = np.empty((len(nodes),len(etas)))

for i,num_nodes in enumerate(nodes):
    for j, eta in enumerate(etas):
        net = mlp.mlp(X_train, Y_train, num_nodes, eta=eta)

        val_err, val_acc, tr_err, tr_acc = net.earlystopping(X_train, Y_train, X_valid, Y_valid, 50)

        ltr_acc[i][j] = tr_acc[-1]
        ltr_err[i][j] = tr_err[-1]

        print("################################")
        print("Test Results")
        test_err[i][j], test_acc[i][j] = net.test(X_test, Y_test)
        print("################################")
        print("Critical Results")
        cri_err[i][j], cri_acc[i][j] =  net.test(X_critical, Y_cri)

plot_data(etas, nodes, ltr_acc)
plot_data(etas, nodes, test_acc)
plot_data(etas, nodes, cri_acc)

#### Testing for the lambda (ReLU scaler) versus eta parameters

def plot_data_lam(x,y,data):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)
    fig.colorbar(cax)

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format( 100*data[j,i])  
            ax.text(x_val, y_val, c, va='center', ha='center', fontsize=16)

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)

    ax.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{ReLU\\ Scaler}$',fontsize=fontsize)

    plt.tight_layout()

    plt.show()

lams = [0.001, 0.01, 0.1, 1]

ltr_acc = np.empty((len(lams),len(etas)))
ltr_err = np.empty((len(lams),len(etas)))
test_acc = np.empty((len(lams),len(etas)))
test_err = np.empty((len(lams),len(etas)))
cri_acc = np.empty((len(lams),len(etas)))
cri_err = np.empty((len(lams),len(etas)))

for i,lam in enumerate(lams):
    for j, eta in enumerate(etas):
        net = mlp.mlp(X_train, Y_train, 10, eta=eta, lam=lam)

        val_err, val_acc, tr_err, tr_acc = net.earlystopping(X_train, Y_train, X_valid, Y_valid, 100)

        ltr_acc[i][j] = tr_acc[-1]
        ltr_err[i][j] = tr_err[-1]

        print("################################")
        print("Test Results")
        test_err[i][j], test_acc[i][j] = net.test(X_test, Y_test)
        print("################################")
        print("Critical Results")
        cri_err[i][j], cri_acc[i][j] =  net.test(X_critical, Y_cri)

plot_data_lam(etas, lams, ltr_acc)
plot_data_lam(etas, lams, test_acc)
plot_data_lam(etas, lams, cri_acc)