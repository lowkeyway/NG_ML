# -*- coding: utf-8 -*-
# @Time    : 2020/12/22 15:53
# @Author  : lowkeyway
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import scipy.optimize as opt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# def gradient(theta, X, y):
#     # return (1/len(X)) * X.T @ (sigmoid(X @ theta) - y)
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
#
#     parameters = int(theta.ravel().shape[1])
#     grad = np.zeros(parameters)
#
#     error = sigmoid(X * theta.T) - y
#     for i in range(parameters):
#         term = np.multiply(error, X[:, i])
#         grad[i] = np.sum(term) / len(X)
#
#     return grad

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad

def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

def data_plot(data):
    sns.lmplot('exam1', 'exam2', hue='admitted', data=data,height=6, fit_reg=False, scatter_kws={"s":50})
    plt.show()



def sigmoid_plot():
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(-10, 10, step=0.01)
    y = sigmoid(x)
    # ax.plot(np.arange(-10, 10, step=0.01), sigmoid(np.arange(-10, 10, step=0.01)))
    ax.plot(x, y)
    ax.set_ylim((-0.1, 1.1))
    ax.set_xlabel('z', fontsize=18)
    ax.set_ylabel('g(z)', fontsize=18)
    ax.set_title("sigmodi function")
    plt.show()

def main_func(argv):
    data = pd.read_csv("ex2data1.txt", names=['exam1', 'exam2', 'admitted'])
    # print(data.head())

    sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))
    data_plot(data)
    sigmoid_plot()

    theta = np.zeros(3)
    data.insert(0, "One", 1)
    cols = data.shape[1]

    # X=np.matrix(data.iloc[:, 0:cols-1].values)
    # Y=np.matrix(data.iloc[:, cols-1:cols].values)

    X = np.array(data.iloc[:, 0:cols - 1].values)
    Y = np.array(data.iloc[:, cols - 1:cols].values)

    # X = np.matrix(X.values)
    # Y = np.matrix(Y.values)
    print("X = ", X)
    print("Y = ", Y)

    c = cost(theta, X, Y)
    print("cost = \n", c)

    g = gradient(theta, X, Y)
    print("gradient = \n", g)

    res = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, Y))
    # res = opt.minimize(fun=cost, x0=theta, args=(X, Y), method="Newton-CG", jac=gradient)
    print("res = \n", res)

    c = cost(res[0], X, Y)
    # c = cost(res.x, X, Y)
    print("Final cost = \n", c)


    # plt.show()


if __name__ == '__main__':
    main_func(sys.argv)