# -*- coding: utf-8 -*-
# @Time    : 2020/11/17 14:29
# @Author  : lowkeyway
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file = "ex1data2.txt"
alpha = 0.01
iters = 1000

def normalEqn(X, Y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ Y
    return theta

def main_func(argv):
    data = pd.read_csv(file, header=None, names=["Size", "Bedrooms", "Price"])
    data = (data - data.mean())/data.std()
    data.insert(0, "ones", 1)

    col = data.shape[1]
    X = data.iloc[:, 0:col-1]
    Y = data.iloc[:, col-1:col]

    X = np.matrix(X.values)
    Y = np.matrix(Y.values)

    print("X = \n", X)
    print("Y = \n", Y)

    theta = normalEqn(X, Y)
    print("theta = \n", theta)


if __name__ == '__main__':
    main_func(sys.argv)