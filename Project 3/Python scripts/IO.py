import pandas as pd
import numpy as np

def loader(filename):

    file = pd.read_csv(filename)
    
    return file

def saver(data, filename):

    data.to_csv(filename)