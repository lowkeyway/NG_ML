# -*- coding: utf-8 -*-
# @Time    : 2020/12/22 15:53
# @Author  : lowkeyway
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

def main_func(argv):
    data = pd.read_csv("ex2data1.txt", names=['exam1', 'exam2', 'admitted'])
    # print(data.head())

    sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))
    sns.lmplot('exam1', 'exam2', hue='admitted', data=data,height=6, fit_reg=False, scatter_kws={"s":50})
    # plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(-10, 10, step=0.01)
    y = sigmoid(x)
    # ax.plot(np.arange(-10, 10, step=0.01), sigmoid(np.arange(-10, 10, step=0.01)))
    ax.plot(x, y)
    ax.set_ylim((-0.1, 1.1))
    ax.set_xlabel('z', fontsize=18)
    ax.set_ylabel('g(z)', fontsize=18)
    ax.set_title("sigmodi function")
    # plt.show()

    theta = np.zeros(3)
    data.insert(0, "One", 1)
    cols = data.shape[1]
    X=np.matrix(data.iloc[:, 0:cols-1].values)
    Y=np.matrix(data.iloc[:, cols-1:cols].values)

    # X = np.matrix(X.values)
    # Y = np.matrix(Y.values)

    c = cost(theta, X, Y)
    print(c)
    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)