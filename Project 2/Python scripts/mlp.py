import numpy as np
from numba import jit

class mlp:

    def __init__(self, inputs, targets, nhidden, batch_size=100, eta=0.1, alpha=0.1, lam=0.1, cost_function="cross-entropy", net_type="classification", act_func="ReLU"):

        self.eta = eta
        self.alpha = alpha
        self.lam = lam
        self.batch_size = batch_size
        self.nhidden = nhidden
        self.ninputs = inputs.shape[1]
        self.ntargets = targets.shape[1]
        self.w = np.sqrt(2/(nhidden+self.ninputs+1)) * (np.random.uniform(-1,1,(nhidden, self.ninputs+1)))
        self.v = np.sqrt(2/(nhidden+self.ntargets+1)) * (np.random.uniform(-1,1,(self.ntargets, nhidden+1)))
        self.cost_function = cost_function
        self.net_type = net_type
        self.act_func = act_func

    def earlystopping(self, inputs, targets, validation, validation_targets, epochs=100):
        
        old_err = float("inf")
        old_acc = 0.0
        best_err = float("inf")
        best_acc = 0.0
        inrow = 0
        val_err = []
        val_acc = []
        best_w = self.w
        best_v = self.v
        tr_err = np.zeros(epochs)
        tr_acc = np.zeros(epochs)
        err = 0
        acc = 0

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
                    acc = r2_valid

                err = self.cal_err(pred_valid, validation_targets, self.cost_function)

                print("The validation error is: ", err)

                if self.net_type == "classification":
                    print("The validation accuracy is: ", acc)
                elif self.net_type == "regression":
                    print("The validation R2-score is: ", r2_valid)
            
                val_err = val_err + [err]
                val_acc = val_acc + [acc]

                if ((err > old_err) and inrow > 50):
                    print("We are stopping early")
                    self.w = best_w
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
                    best_w = self.w
                    best_v = self.v
                    best_err = err
                    old_err = err
                    if self.net_type == "classification":
                        best_acc = acc
                        old_acc = acc

        self.w = best_w
        self.v = best_v    
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
        
        # Forward pass from inputs to the hidden layer
        in_w_b = np.c_[np.ones((len(inputs))),inputs]
        self.hidden_layer = in_w_b @ self.w.T 
        if self.act_func == "ReLU":
            self.hidden_layer = self.relu(self.hidden_layer, self.lam)
        elif self.act_func == "LReLU":
            self.hidden_layer = self.lrelu(self.hidden_layer, self.alpha, self.lam)
        elif self.act_func == "ELU":
            self.hidden_layer = self.elu(self.hidden_layer, self.alpha)
        elif self.act_func == "SELU":
            self.hidden_layer = self.selu(self.hidden_layer, self.alpha, self.lam)
        elif self.act_func == "sigmoid":
            self.hidden_layer = self.sigmoid(self.hidden_layer)
        elif self.act_func == "tanh":
            self.hidden_layer = np.tanh(self.hidden_layer)

        if(np.NaN in self.hidden_layer):
            print("Error in the ReLU")
            input("continue")

        # Forward pass from hidden layer to the output layer
        h_w_b = np.c_[np.ones((len(self.hidden_layer))), self.hidden_layer]
        pred = h_w_b @ self.v.T


        return pred

    @jit
    def backwards_regression(self, inputs, predicted, targets):
        
        # Calculating deltas
        delta_W = np.zeros(self.w.shape)
        delta_V = np.zeros(self.v.shape)
        der = 1

        for i in range(predicted.shape[0]):

            ac_pred = predicted[i].reshape(len(predicted[i]), 1)
            ac_true = targets[i].reshape(len(targets[i]), 1)
            ac_hid_b = np.append(1, self.hidden_layer[i])
            ac_hid_b = ac_hid_b.reshape(1, len(ac_hid_b))

            delta_V = delta_V - (ac_true - ac_pred) @ ac_hid_b
            sigma = np.sum((ac_true - ac_pred)*self.v[:,1:], axis=0).reshape(self.v.shape[1]-1,1)

            ac_hid = self.hidden_layer[i].reshape(len(self.hidden_layer[i]),1)
            ac_in_b = np.append(1, inputs[i])
            ac_in_b = ac_in_b.reshape(1, len(ac_in_b))

            if self.act_func == "ReLU":
                der = self.der_relu(ac_hid, self.lam)
            elif self.act_func == "LReLU":
                der = self.der_lrelu(ac_hid, self.alpha, self.lam)
            elif self.act_func == "ELU":
                der = self.der_elu(ac_hid, self.alpha)
            elif self.act_func == "SELU":
                der = self.der_selu(ac_hid, self.alpha, self.lam)
            elif self.act_func == "sigmoid":
                der = self.der_sig(ac_hid)
            elif self.act_func == "tanh":
                der = self.der_tanh(ac_hid)
            delta_W = delta_W - sigma * der @ ac_in_b
        
        #Updating

        self.w = self.w - self.eta * (1/predicted.shape[0]) * delta_W
        self.v = self.v - self.eta * (1/predicted.shape[0]) * delta_V

    
    def forward_classification(self, inputs):
        
        # Forward pass from inputs to the hidden layer
        in_w_b = np.c_[np.ones((len(inputs))),inputs]
        self.hidden_layer = in_w_b @ self.w.T 
        if self.act_func == "ReLU":
            self.hidden_layer = self.relu(self.hidden_layer, self.lam)
        elif self.act_func == "LReLU":
            self.hidden_layer = self.lrelu(self.hidden_layer, self.alpha, self.lam)
        elif self.act_func == "ELU":
            self.hidden_layer = self.elu(self.hidden_layer, self.alpha)
        elif self.act_func == "SELU":
            self.hidden_layer = self.selu(self.hidden_layer, self.alpha, self.lam)
        elif self.act_func == "sigmoid":
            self.hidden_layer = self.sigmoid(self.hidden_layer)
        elif self.act_func == "tanh":
            self.hidden_layer = np.tanh(self.hidden_layer)

        if(np.NaN in self.hidden_layer):
            print("Error in the ReLU")
            input("continue")

        # Forward pass from hidden layer to the output layer
        h_w_b = np.c_[np.ones((len(self.hidden_layer))), self.hidden_layer]
        pred = h_w_b @ self.v.T

        pred = self.softmax(pred)
        if(np.NaN in pred):
            print("Error in the softmax")
            input("continue")

        return pred

    def backwards_classification(self, inputs, predicted, targets):
        
        # Calculating deltas
        delta_W = np.zeros(self.w.shape)
        delta_V = np.zeros(self.v.shape)
        der = 1

        for i in range(predicted.shape[0]):

            ac_pred = predicted[i].reshape(len(predicted[i]), 1)
            ac_true = targets[i].reshape(len(targets[i]), 1)
            ac_hid_b = np.append(1, self.hidden_layer[i])
            ac_hid_b = ac_hid_b.reshape(1, len(ac_hid_b))

            delta_V = delta_V - (ac_true - ac_pred) @ ac_hid_b
            sigma = np.sum((ac_true - ac_pred)*self.v[:,1:], axis=0).reshape(self.v.shape[1]-1,1)

            ac_hid = self.hidden_layer[i].reshape(len(self.hidden_layer[i]),1)
            ac_in_b = np.append(1, inputs[i])
            ac_in_b = ac_in_b.reshape(1, len(ac_in_b))

            if self.act_func == "ReLU":
                der = self.der_relu(ac_hid, self.lam)
            elif self.act_func == "LReLU":
                der = self.der_lrelu(ac_hid, self.alpha, self.lam)
            elif self.act_func == "ELU":
                der = self.der_elu(ac_hid, self.alpha)
            elif self.act_func == "SELU":
                der = self.der_selu(ac_hid, self.alpha, self.lam)
            elif self.act_func == "sigmoid":
                der = self.der_sig(ac_hid)
            elif self.act_func == "tanh":
                der = self.der_tanh(ac_hid)
            delta_W = delta_W - sigma * der @ ac_in_b
        
        #Updating

        self.w = self.w - self.eta * (1/predicted.shape[0]) * delta_W
        self.v = self.v - self.eta * (1/predicted.shape[0]) * delta_V


    def test(self, test, test_targets):
        
        if self.net_type == "classification":
            pred = self.forward_classification(test)
            acc = self.cal_acc(pred, test_targets)
            conf = self.confusion_table(pred, test_targets)
        elif self.net_type == "regression":
            pred = self.forward_regression(test)
            r2 = self.cal_r2(pred, test_targets)

        err = self.cal_err(pred, test_targets, self.cost_function)

        print("The test error is: ", err)

        if self.net_type == "classification":
            print("The test accuracy is: ", acc)
            print("Confusion matrix:")
            print(conf)
            return err, acc
        elif self.net_type == "regression":
            print("The test R2-score is: ", r2)
            return err, r2
    
    def cal_err(self, predicted, targets, costf):

        if costf == "cross-entropy":
            err = -np.sum(targets * np.log(predicted))
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

    #Activation functions

    @jit
    def softmax(self, x):

        for i in range(len(x)):
            div = np.sum(np.exp(x[i]))
            x[i] = np.exp(x[i]) / div
        
        return x
    def relu(self, x, lam):

        x = lam * x
        x[x <= 0] = 0

        return x

    def der_relu(self,x, lam):

        x[x > 0] = lam

        return x

    def lrelu(self, x, alpha, lam):

        x = lam * x
        x[x <= 0] = alpha * x[x <= 0]

        return x
    
    def der_lrelu(self, x, alpha, lam):

        x[x > 0] = lam
        x[x <= 0] = lam * alpha

        return x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def der_sig(self, x):

        return x * (1 - x)

    def der_tanh(self, x):

        return 1 - x**2

    def elu(self, x, alpha):

        x[x <= 0] = alpha * (np.exp(x[x<=0]) - 1)

        return x

    def der_elu(self, x, alpha):

        x[x > 0] = 1
        x[x <= 0] = x[x <= 0] + alpha

        return x

    def selu(self, x, alpha, lam):

        x[x <= 0] = alpha * (np.exp(x[x<=0]) - 1)
        x = lam * x

        return x
    
    def der_selu(self, x, alpha, lam):

        x[x > 0] = lam
        x[x <= 0] = x[x <= 0] + (alpha * lam)

        return x
    
