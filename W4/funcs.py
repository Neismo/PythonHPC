import numpy as np

def standardize_rows(data, mean, std):
    for i in range(data.shape[0]):
        data[i, :] = (data[i, :] - mean) / std
    return data

def outer(x, y):
    return x[:, None] * y

def distmat_1d(x, y):
    return abs((x - y[:, None]).T)