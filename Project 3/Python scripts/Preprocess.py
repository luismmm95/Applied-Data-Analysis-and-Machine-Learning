import pandas as pd
import numpy as np
from numba import jit

def rem_col(df, col_name):

    df_mod = df.drop(col_name, axis=1)

    return df_mod

@jit
def to_category(df, col_name):

    cats = []

    to_class = df[col_name]

    for item in df[col_name]:

        if item not in cats:
            cats = cats + [item]
    
    if len(cats) != 2:

        for i in range(to_class.size):

            ind = cats.index(to_class[i])

            to_class[i] = ind+1
    
    else:

        for i in range(to_class.size):

            if cats.index(to_class[i]) == 0:
                to_class[i] = -1
            else:
                to_class[i] = 1

    df_mod = df
    df_mod[col_name] = to_class

    return df_mod, cats

@jit
def bin_class_miss(df, col_name):

    df_mod = df

    to_mod = df[col_name]

    for i in range(df_mod[col_name].size):

        if df[col_name].isnull()[i]:
            to_mod[i] = -1
        else:
            to_mod[i] = 1

    df[col_name] = to_mod
    
    return df_mod

@jit
def miss_to_num(df, col_name, num=0):

    df_mod = df

    to_mod = df[col_name]

    for i in range(df_mod[col_name].size):

        if df[col_name].isnull()[i]:
            to_mod[i] = num

    df[col_name] = to_mod
    
    return df_mod

def create_label_data_split(df, label_name):

    if(label_name != None):
        temp_out = df[label_name]
        temp_in = df.drop(label_name, axis=1)

        outputs = temp_out.values
        inputs = temp_in.values
        return inputs, outputs
    else:
        inputs = df.values
        return inputs

@jit
def make_categorical(x, num_cat, legend=None):

    labels = np.zeros((len(x), num_cat))
    if legend == None:
        w_class = []
    else:
        w_class = legend

    for i, label in enumerate(x):

        if label in w_class:
            ind = w_class.index(label) 
        else:
            w_class = w_class + [label]
            ind = len(w_class) - 1

        labels[i, ind] = 1

    return labels, w_class

def decode_cat(x, legend, l=False):
    
    if l:
        decoded = []

        for i in range(len(x)):
            decoded = decoded + [legend[np.argmax(x[i])]]
    else:
        decoded = np.zeros(len(x))

        for i in range(len(x)):
            decoded[i] = legend[np.argmax(x[i])]
    
    return decoded

def test_train_split(data, labels, test_per=0.4):

    per = np.random.permutation(len(data))

    data_per = data[per]
    labels_per = labels[per]

    num_test = int(np.round(test_per * len(data)))

    test = data_per[:num_test]
    test_label = labels_per[:num_test]
    train = data_per[num_test:]
    train_label = labels_per[num_test:]

    return train, train_label, test, test_label, per, num_test

def guess_image_dim(in_shape):
    side_len = int(np.sqrt(in_shape))
    if np.abs(in_shape-side_len*side_len)<2:
        return (int(side_len), int(side_len), 1)
    else:
        side_len = int(np.sqrt(in_shape/3))
        return (side_len, side_len, 3)

def resize(data):

    resize_dim = guess_image_dim(data.shape[1])
    print(resize_dim)
    print(data.shape[0])

    data_resize = np.empty((data.shape[0], resize_dim))

    for i, img in enumerate(data):
        data_resize[i] = np.resize(img, resize_dim)
    
    return data_resize

def add_column(data, new_col, name, loc):
    
    n_data = data
    n_data.insert(loc, name, new_col)

    return n_data