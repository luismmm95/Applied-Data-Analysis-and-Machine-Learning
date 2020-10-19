import pandas as pd
import numpy as np
import Neural as nn
import IO
import Preprocess
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn

np.set_printoptions(suppress=True)

raw_data = IO.loader("hmnist_8_8_RGB.csv")

raw_metadata = IO.loader("HAM10000_metadata.csv")

data, raw_labels = Preprocess.create_label_data_split(raw_data, "label")

labels, legend = Preprocess.make_categorical(raw_labels, 7)

data = data.astype(np.float64)

train, train_labels, test, test_labels, per, num_test = Preprocess.test_train_split(data, labels, 0.25)

raw_test_metadata = raw_metadata.take(per[:num_test])

train, train_labels, valid, valid_labels, _, _ = Preprocess.test_train_split(train, train_labels, 0.1)

filenames = ["one_hidden_layer.png", "two_hidden_layers.png", "three_hidden_layers.png", "four_hidden_layers.png", "five_hidden_layers.png"]
filenames2 = ["1h.csv", "2h.csv", "3h.csv", "4h.csv", "5h.csv"]
layer_size = [128, 64, 32, 16, 16]
lams = [0.1, 1, 1, 1, 1]
w_lg = []
for i in range(len(legend)):
    if legend[i] == 0:
        w_lg = w_lg + ["akiec"]
    elif legend[i] == 1:
        w_lg = w_lg + ["bcc"]
    elif legend[i] == 2:
        w_lg = w_lg + ["bkl"]
    elif legend[i] == 3:
        w_lg = w_lg + ["df"]
    elif legend[i] == 4:
        w_lg = w_lg + ["nv"]
    elif legend[i] == 5:
        w_lg = w_lg + ["vasc"]
    elif legend[i] == 6:
        w_lg = w_lg + ["mel"]

#### Finding the best number of hidden layers

f = open("results.txt", mode="a")

for i in range(len(filenames)):
    net = nn.Mlp(train, train_labels, batch_size=128, eta=1.0e-3)

    for j in range(i+1):
        net.add_layer(layer_size[j], lams[j])

    net.init_weights()

    net.earlystopping(train, train_labels, valid, valid_labels, 1000, 100)

    err, acc, conf = net.test(test, test_labels, True, filenames[i], legend)
    f.write("Results for %d hidden layers \n Error: %.04f \n Accuracy: %.04f\n Confusion matrix: %s\n" % ((i+1), err, acc, conf))
    
    pred = net.predict(test)
    predict = Preprocess.decode_cat(pred, w_lg, l=True)
    mod_test_data = Preprocess.add_column(raw_test_metadata, predict, "Predicted dx", 3)

    IO.saver(mod_test_data, filenames2[i])

    raw_test_metadata = Preprocess.rem_col(raw_test_metadata, "Predicted dx")

f.close()

#### Testing for the lambda (ReLU scaler) versus eta parameters

def plot_data_lam(x,y,data,filename):

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

    plt.savefig(filename)
    plt.clf()

### Trying to find the best hyperparameters

f1 = open("results2.txt", mode="a")

lams = [[0.1, 0.1, 0.1], [0.1, 0.1, 1], [0.1, 1, 0.1], [0.1, 0.01, 0.01], [0.1, 1, 1]]
etas = [1e-6, 1e-5, 1e-4, 1e-3]
lam_label = ["equal", "last", "middle", "first", "last_two"]

accs = np.empty((len(lams), len(etas)))

for i, lam in enumerate(lams):
    for j, eta in enumerate(etas):
        net = nn.Mlp(train, train_labels, batch_size=128, eta=eta)

        net.add_layer(128, lam[0])
        net.add_layer(64, lam[1])
        net.add_layer(32, lam[2])

        net.init_weights()

        net.earlystopping(train, train_labels, valid, valid_labels, 1000, 100)

        err, acc, conf = net.test(test, test_labels)
        f1.write("Results for eta: %.06f and lam type: %s \n Error: %.04f \n Accuracy: %.04f\n Confusion matrix: %s\n" % (eta, lam_label[i], err, acc, conf))
        
        accs[i,j] = acc

        #### Uncomment to create metadata files for these tests

        # pred = net.predict(test)
        # predict = Preprocess.decode_cat(pred, w_lg, l=True)
        # mod_test_data = Preprocess.add_column(raw_test_metadata, predict, "Predicted dx", 3)

        # filename = "lam_%s_eta_%.06f.csv" % (lam_label[i], eta)

        # IO.saver(mod_test_data, filename)

        # raw_test_metadata = Preprocess.rem_col(raw_test_metadata, "Predicted dx")

f1.close()

plot_data_lam(etas, lam_label, accs, "acc_lam_eta.png")

#### Confusion matrix and metadata for best model

net = nn.Mlp(train, train_labels, batch_size=128, eta=0.001)

net.add_layer(128, lam=0.1)
net.add_layer(64, 1)
net.add_layer(32, 0.1)

net.init_weights()

net.earlystopping(train, train_labels, valid, valid_labels, 1000, 50)

net.test(test, test_labels, True, "conf_mat_best.png", legend)

pred = net.predict(test)
predict = Preprocess.decode_cat(pred, w_lg, l=True)
mod_test_data = Preprocess.add_column(raw_test_metadata, predict, "Predicted dx", 3)

IO.saver(mod_test_data, "best_pred.csv")

raw_test_metadata = Preprocess.rem_col(raw_test_metadata, "Predicted dx")
