# -*- coding: utf-8 -*-
# @Time    : 2020/11/16 21:00
# @Author  : lowkeyway
import sys
import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

file = "ex1data2.txt"
alpha = 0.01
iters = 1000

def computeCost(X, Y, thelta):
    error = X * thelta.T - Y
    inner = np.power(error, 2)
    return np.sum(inner)/ (2 * len(X))

def gradientDescent(X, Y, thelta, alpha, iters):
    parameters = X.shape[1]
    # temp = np.matrix(np.zeros(thelta.shape))
    temp = np.matlib.zeros((1, parameters))
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * thelta.T) - Y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = thelta[0, j] - ((alpha / len(X)) * np.sum(term))

        thelta = temp
        cost[i] = computeCost(X, Y, thelta)

    return thelta, cost

def main_func(argv):
    data = pd.read_csv(file, header=None, names=["Size", "Bedrooms", "Price"])
    print("\n========== data.head ==========\n", data.head())

    data = (data - data.mean()) / data.std()
    print("\n========== data.head ==========\n", data.head())

    data.insert(0, "Ones", 1)
    print("\n========== data.head ==========\n", data.head())

    col = data.shape[1]
    X = data.iloc[:, 0 : col - 1]
    Y = data.iloc[:, col - 1 : col]
    theta = np.matrix(np.array([0, 0, 0]))
    # theta = np.matlib.zeros((1, 3))

    X = np.matrix(X.values)
    Y = np.matrix(Y.values)

    theta, cost = gradientDescent(X, Y, theta, alpha, iters)

    cost = computeCost(X, Y, theta)

    print("Theta = ", theta)
    print("Cost = ", cost)




if __name__ == '__main__':
    main_func(sys.argv)