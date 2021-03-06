# -*- coding: utf-8 -*-
# @Time    : 2020/11/15 16:21
# @Author  : lowkeyway

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

alpha = 0.01
iters = 1000
BLOCK = 10

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def pltShowThetaLine(data, theta, index):
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predict Profit vs. Population Size ( ' + str(index) + " )")
    f = theta[0, 0] + (theta[0, 1] * x)
    ax.plot(x, f, 'r', label='Prediction')

    plt.show()

def pltShowCostLine(alpha, cost):
    index = cost.shape[0]

    x = np.arange(0, index, 1)
    y = cost

    plt.figure("Alpha = " + str(alpha))
    plt.title("Alpha Cost Line")
    plt.xlabel("Item")
    plt.ylabel("Cost")
    plt.plot(x, y)
    plt.show()


def gradientDescent(data, X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y
        if(i % (iters / BLOCK) == 0):
            pltShowThetaLine(data, theta, i)

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - (alpha / len(X)) * np.sum(term)

        theta = temp
        cost[i] =  computeCost(X, y, theta)

    return theta, cost

def main_func(argv):
    path = 'ex1data1.txt'
    data = pd.read_csv(path, header=None, names = ['Population', 'Profit'])
    print("\n============data.head============\n", data.head())
    print("\n============data.describe============\n", data.describe())

    #data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
    #plt.show()

    data.insert(0, 'Ones', 1)
    print("\n============data.head============\n", data.head())
    print("\n============data.describe============\n", data.describe())

    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1: cols]

    print("\n============X.head============\n", X.head())
    print("\n============y.head============\n", y.head())

    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0]))

    print("\n============X============\n", X)
    print("\n============y============\n", y)
    print("\n============theta============\n", theta)

    cost = computeCost(X, y, theta)
    print("\n============cost============\n", cost)

    g, cost = gradientDescent(data, X, y, theta, alpha, iters)
    print("\n============theta============\n", g)
    print("\n============Final Cost============\n", cost[iters - 1])


    pltShowThetaLine(data, g, iters)

    pltShowCostLine(alpha, cost)


if __name__ == '__main__':
    main_func(sys.argv)