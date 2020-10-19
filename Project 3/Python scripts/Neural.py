import numpy as np
from numba import jit
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

def im2col(A, BSZ, stepsize=1):
    """
    Replica of im2col function in MatLab, taken from stackoverflow: https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python/30110497
    """
    # Parameters
    m,n,l = A.shape
    s0, s1, s2 = A.strides    
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    ndeps = l-BSZ[2]+1
    shp = BSZ[0],BSZ[1],BSZ[2],nrows,ncols,ndeps
    strd = s0,s1,s2,s0,s1,s2

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1]*BSZ[2],-1)[:,::stepsize]

def im2row(A, BSZ, stepsize=1):

    return im2col(A, BSZ, stepsize).T


class Mlp:

    def __init__(self, inputs, targets, batch_size=100, eta=0.1, alpha=0.1, lam=0.1, cost_function="cross-entropy", net_type="classification", act_func="ReLU"):

        self.eta = eta
        self.alpha = alpha
        self.lam = lam
        self.batch_size = batch_size
        self.ninputs = inputs.shape[1]
        self.ntargets = targets.shape[1]
        self.cost_function = cost_function
        self.net_type = net_type
        self.act_func = ActivationFunction(act_func)

        self.hidden_layers = []

    def add_layer(self, n_nodes, lam=0.1, alpha=0.1, act_func="ReLU"):

        hid = Layer(n_nodes, lam, alpha, act_func)
        self.hidden_layers = self.hidden_layers + [hid]
    
    def init_weights(self):

        n_inputs = self.ninputs
        for hidden_layer in self.hidden_layers:
            hidden_layer.init_weights(n_inputs)
            n_inputs = hidden_layer.n_nodes

        self.v = np.sqrt(2/(self.ntargets+n_inputs+1)) * (np.random.uniform(-1,1,(self.ntargets, n_inputs+1)))

    def earlystopping(self, inputs, targets, validation, validation_targets, epochs=100, stop_num=50):
        
        old_err = float("inf")
        best_err = float("inf")
        best_acc = 0.0
        best_err_tr = float("inf")
        best_acc_tr = 0.0
        inrow = 0
        val_err = []
        val_acc = []
        best_v = self.v
        tr_err = np.zeros(epochs)
        tr_acc = np.zeros(epochs)

        for i in range(epochs):
            print("Epoch ", i)

            self.train(inputs, targets, 1)
            if self.net_type == "classification":
                pred_train = self.forward_classification(inputs)
                tr_acc[i] = self.cal_acc(pred_train, targets)
            elif self.net_type == "regression":
                pred_train = self.forward_regression(inputs)
                r2_train = self.cal_r2(pred_train, targets)
                tr_acc[i] = r2_train
            tr_err[i] = self.cal_err(pred_train, targets, self.cost_function)

            print("The training error is: ", tr_err[i])
            if self.net_type == "classification":
                print("The training accuracy is: ", tr_acc[i])
            elif self.net_type == "regression":
                print("The training r2-score is: ", r2_train)

            if i % 1 == 0:
                if self.net_type == "classification":
                    pred_valid = self.forward_classification(validation)
                    pred_train = self.forward_classification(inputs)
                    acc = self.cal_acc(pred_valid, validation_targets)
                elif self.net_type == "regression":
                    pred_valid = self.forward_regression(validation)
                    pred_train = self.forward_regression(inputs)
                    r2_valid = self.cal_r2(pred_valid, validation_targets)
                    acc= r2_valid

                err = self.cal_err(pred_valid, validation_targets, self.cost_function)

                print("The validation error is: ", err)

                if self.net_type == "classification":
                    print("The validation accuracy is: ", acc)
                elif self.net_type == "regression":
                    print("The validation R2-score is: ", r2_valid)
            
                val_err = val_err + [err]
                val_acc = val_acc + [acc]

                if ((err > old_err) and inrow > stop_num):
                    print("We are stopping early")
                    for hidden_layer in self.hidden_layers:
                        hidden_layer.optimize()
                    self.v = best_v
                    break
                if (err > old_err):
                    print("Case 1: The error is getting worst")
                    inrow += 1
                    old_err = err
                    if self.net_type == "classification":
                        old_acc = acc
                elif (err > best_err):
                    print("Case 2: The error isn't getting worst, but not the best error")
                    old_err = err
                    if self.net_type == "classification":
                        old_acc = acc
                else:
                    print("Case 3: The error is lowering")
                    inrow = 0
                    for hidden_layer in self.hidden_layers:
                        hidden_layer.set_best_weight()
                    best_v = self.v
                    best_err = err
                    old_err = err
                    best_err_tr = tr_err[i]
                    best_acc = acc
                    best_acc_tr = tr_acc[i]
                    if self.net_type == "classification":
                        best_acc = acc
                        old_acc = acc

        for hidden_layer in self.hidden_layers:
            hidden_layer.optimize()
        self.v = best_v
        print("########################################")
        print("Best validation error:", best_err)
        print("Best training error:", best_err_tr)
        print("Best validation accuracy:", best_acc)
        print("Best training accuracy:", best_acc_tr)    
        return val_err, val_acc, tr_err, tr_acc
    
    @jit
    def train(self, inputs, targets, iterations=100):
        
        size_batch = 100
        num_batches = int(np.floor(len(inputs) / size_batch))

        for i in range(iterations):
            per = np.random.permutation(len(inputs))
            for j in range(num_batches):
                dat = inputs[per[size_batch*j:size_batch*(j+1)]]
                tar = targets[per[size_batch*j:size_batch*(j+1)]]
                if self.net_type == "classification":
                    pred = self.forward_classification(dat)
                    self.backwards_classification(dat, pred, tar)
                elif self.net_type == "regression":
                    pred = self.forward_regression(dat)
                    self.backwards_regression(dat, pred, tar)
    
    def forward_regression(self, inputs):
        
        # Forward pass from inputs to the hidden layer and hidden layer to hidden layer
        inp = inputs
        for hidden_layer in self.hidden_layers:
            hidden_layer.forward(inp)
            inp = hidden_layer.nodes

        # Forward pass from hidden layer to the output layer
        h_w_b = np.c_[np.ones((len(inp))), inp]
        pred = h_w_b @ self.v.T

        return pred

    def backwards_regression(self, inputs, predicted, targets):
        
        delta_V = np.zeros(self.v.shape)
        sigma = np.empty(predicted.shape[0])

        for i in range(predicted.shape[0]):

            ac_pred = predicted[i].reshape(len(predicted[i]), 1)
            ac_true = targets[i].reshape(len(targets[i]), 1)
            ac_hid_b = np.append(1, self.hidden_layers[-1].nodes[i])
            ac_hid_b = ac_hid_b.reshape(1, len(ac_hid_b))

            delta_V = delta_V - (ac_true - ac_pred) @ ac_hid_b
            sigma[i] = np.sum((ac_true - ac_pred)*self.v[:,1:], axis=0).reshape(self.v.shape[1]-1,1)

        num_hid_lay = len(self.hidden_layers) - 2
        inp = self.hidden_layers[-2].nodes
        for i, hidden_layer in enumerate(reversed(self.hidden_layers)):
            sigma = hidden_layer.calc_delta(inp, sigma)
            ind = num_hid_lay - i
            if ind >= 0:
                inp = self.hidden_layers[ind].nodes
            else:
                inp = inputs

        
        #Updating

        for hidden_layer in self.hidden_layers:
            hidden_layer.update_weights(self.eta)

        self.v = self.v - self.eta * (1/predicted.shape[0]) * delta_V
    
    def forward_classification(self, inputs):
        
        # Forward pass from inputs to the hidden layer
        inp = inputs
        for hidden_layer in self.hidden_layers:
            hidden_layer.forward(inp)
            inp = hidden_layer.nodes

        # Forward pass from hidden layer to the output layer
        h_w_b = np.c_[np.ones((len(inp))), inp]
        pred = h_w_b @ self.v.T

        pred = self.softmax(pred)
        if(np.NaN in pred):
            print("Error in the softmax")
            input("continue")

        return pred

    def backwards_classification(self, inputs, predicted, targets):
        
        delta_V = np.zeros(self.v.shape)
        sigma = np.empty((predicted.shape[0], self.v.shape[1]-1, 1))

        for i in range(predicted.shape[0]):

            ac_pred = predicted[i].reshape(len(predicted[i]), 1)
            ac_true = targets[i].reshape(len(targets[i]), 1)
            ac_hid_b = np.append(1, self.hidden_layers[-1].nodes[i])
            ac_hid_b = ac_hid_b.reshape(1, len(ac_hid_b))

            delta_V = delta_V - (ac_true - ac_pred) @ ac_hid_b
            sigma[i] = np.sum((ac_true - ac_pred)*self.v[:,1:], axis=0).reshape(self.v.shape[1]-1,1)

        num_hid_lay = len(self.hidden_layers) - 2
        if num_hid_lay >= 0:
            inp = self.hidden_layers[-2].nodes
        else:
            inp = inputs
        for i, hidden_layer in enumerate(reversed(self.hidden_layers)):
            sigma = hidden_layer.calc_delta(inp, sigma)
            ind = num_hid_lay - (i+1)
            if ind >= 0:
                inp = self.hidden_layers[ind].nodes
            else:
                inp = inputs

        
        #Updating

        for hidden_layer in self.hidden_layers:
            hidden_layer.update_weights(self.eta)

        self.v = self.v - self.eta * (1/predicted.shape[0]) * delta_V

    def test(self, test, test_targets, pdconf=False, filename="", legend=None):
        
        if self.net_type == "classification":
            pred = self.forward_classification(test)
            acc = self.cal_acc(pred, test_targets)
            conf = self.confusion_table(pred, test_targets)
            if pdconf:
                temp_pred = self.predict(test)
                if legend != None:
                    predict = np.empty(len(temp_pred))
                    targets = np.empty(len(test_targets))
                    for i in range(len(targets)):
                        predict[i] = legend[np.argmax(temp_pred[i])]
                        targets[i] = legend[np.argmax(test_targets[i])]
                confus = ConfusionMatrix(targets, predict, display_sum=True)
        elif self.net_type == "regression":
            pred = self.forward_regression(test)
            r2 = self.cal_r2(pred, test_targets)

        err = self.cal_err(pred, test_targets, self.cost_function)

        print("The test error is: ", err)

        if self.net_type == "classification":
            print("The test accuracy is: ", acc)
            print("Confusion matrix:")
            print(conf)
            if pdconf:
                confus.plot(backend="seaborn")
                plt.savefig(filename)
                plt.clf()
            return err, acc, conf
        elif self.net_type == "regression":
            print("The test R2-score is: ", r2)
            return err, r2
        
    def predict(self, to_pred):

        if self.net_type == "classification":
            pred = self.forward_classification(to_pred)
            prediction = np.zeros((len(pred),len(pred[0])))
            for i in range(len(pred)):
                prediction[i,np.argmax(pred[i])] = 1
        elif self.net_type == "regression":
            pred = self.forward_regression(to_pred)

        return prediction
    
    def cal_err(self, predicted, targets, costf):

        if costf == "cross-entropy":
            err = -np.sum(targets * np.log(predicted + 1e-10))
        elif costf == "squared-error":
            err = np.sum((predicted - targets)**2)
        elif costf == "MSE":
            err = (1/len(targets))*np.sum((predicted - targets)**2)

        return err

    def cal_acc(self, predicted, targets):

        corr = 0
        for i in range(len(predicted)):
            if np.argmax(predicted[i]) == np.argmax(targets[i]):
                corr += 1
        
        return corr / len(predicted)
    
    def cal_r2(self, predicted, targets):

        mu = np.mean(targets)
        SS_tot = np.sum((targets - mu)**2)
        SS_res = np.sum((targets - predicted)**2)
        
        r2 = 1 - (SS_res/SS_tot)

        return r2

    
    def confusion_table(self, predicted, targets):

        conf = np.zeros((len(predicted[0]), len(predicted[0])))

        for i in range(len(predicted)):
            conf[np.argmax(predicted[i]), np.argmax(targets[i])] += 1
        
        return conf

    @jit
    def softmax(self, x):

        for i in range(len(x)):
            div = np.sum(np.exp(x[i]))
            x[i] = np.exp(x[i]) / div
        return x

class Layer:

    def __init__(self, n_nodes, lam=0.1, alpha=0.1, act_func="ReLU"):

        self.n_nodes = n_nodes
        self.lam = lam
        self.alpha = 0.1
        self.act_func = ActivationFunction(act_func)
    
    def init_weights(self, n_inputs):

        self.n_inputs = n_inputs
        self.w = np.sqrt(2/(self.n_nodes+self.n_inputs+1)) * (np.random.uniform(-1,1,(self.n_nodes, self.n_inputs+1)))
    
    def forward(self, inputs):

        in_w_b = np.c_[np.ones((len(inputs))),inputs]
        self.nodes = in_w_b @ self.w.T
        self.nodes = self.act_func.evaluate(self.nodes, self.alpha, self.lam)

    def calc_delta(self, inputs, err):

        self.delta = np.zeros(self.w.shape)
        sigma = np.zeros((len(self.nodes), self.w.shape[1]-1, 1))

        for i in range(len(self.nodes)):

            sigma[i] = np.sum(err[i]*self.w[:,1:], axis=0).reshape(self.w.shape[1]-1,1)

            ac_layer = self.nodes[i].reshape(len(self.nodes[i]),1)
            prev_layer_b = np.append(1, inputs[i])
            prev_layer_b = prev_layer_b.reshape(1, len(prev_layer_b))

            der = self.act_func.eval_der(ac_layer, self.alpha, self.lam)
            self.delta = self.delta - err[i] * der @ prev_layer_b

        return sigma

    def update_weights(self, eta):

        self.w = self.w - eta * (1/len(self.nodes)) * self.delta

    def set_best_weight(self):

        self.best_w = self.w
    
    def optimize(self):

        self.w = self.best_w

class ActivationFunction:

    def __init__(self, func):

        self.func = func

    def evaluate(self, x, alpha, lam):

        if self.func == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.func == "tanh":
            return np.tanh(x)
        elif self.func == "ReLU":
            x = lam * x
            x[x <= 0] = 0
            return x
        elif self.func == "LReLU":
            x = lam * x
            x[x <= 0] = alpha * x[x <= 0]
            return x
        elif self.func == "ELU":
            x[x > 0] = 1
            x[x <= 0] = alpha * (np.exp(x[x <= 0]) - 1)
            return x
        elif self.func == "SELU":
            x[x <= 0] = alpha * (np.exp(x[x <= 0]) - 1)
            x = lam * x
            return x
        
    def eval_der(self, x, alpha, lam):

        if self.func == "sigmoid":
            return x * (1 - x)
        elif self.func == "tanh":
            return 1 - x**2
        elif self.func == "ReLU":
            x[x > 0] = lam
            return x
        elif self.func == "LReLU":
            x[x > 0] = lam
            x[x <= 0] = lam * alpha
            return x
        elif self.func == "ELU":
            x[x > 0] = 1
            x[x <= 0] = x[x <= 0] + alpha
            return x
        elif self.func == "SELU":
            x[x <= 0] = lam * (x[x <= 0] + alpha) 
            x[x > 0] = lam
            return x

class ConvLayer:

    def __init__(self, num_filter, rf, depth, pad_size, stride):

        self.num_filter = num_filter
        self.rf = rf
        self.depth = depth
        self.pad_size = pad_size
        self.stride = stride
    
    def init_filter(self):

        n = self.rf * self.rf * self.depth

        self.filters = np.random.randn(self.num_filter, self.rf, self.rf, self.depth) * np.sqrt(2.0/n)

        self.bias = np.zeros(self.num_filter)
    
    def zero_pad(self, img):

        n, m = img.shape[0:2]

        padded_img = np.zeros((n+2*self.pad_size, m+2*self.pad_size, img.shape[2]))

        padded_img[self.pad_size:n+self.pad_size, self.pad_size:m+self.pad_size, :] = img

        return padded_img, n, m

    def conv(self, img, n, m, filters):

        img_col = im2col(img, filters.shape[1:], self.stride)
        filters_row = im2row(filters, filters.shape[1:], self.stride)

        conv_img = (filters_row @ img_col).T

        n2 = (n - self.rf + 2 * self.pad_size)/self.stride + 1
        m2 = (n - self.rf + 2 * self.pad_size)/self.stride + 1

        conv_img = conv_img.resize(n2, m2, len(filters))

        self.out = conv_img

        return conv_img
    
    def forward(self, img):

        imgp, n, m = self.zero_pad(img)

        return self.conv(imgp, n, m, self.filters)
    
    def backpropagation(self, img):
        pass
