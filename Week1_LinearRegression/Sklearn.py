# -*- coding: utf-8 -*-
# @Time    : 2020/11/16 23:09
# @Author  : lowkeyway
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

alpha = 0.01
iters = 1000
file = "ex1data1.txt"

def main_func(argv):
    data = pd.read_csv(file, names=['Population', 'Profit'])
    print("\n========== data.head ==========\n", data.head() )

    data.insert(0, "Ones", 1)
    print("\n========== data.head ==========\n", data.head() )

    col = data.shape[1]
    X = data.iloc[:, 0 : col - 1]
    Y = data.iloc[:, col - 1 : col]

    X = np.matrix(X.values)
    Y = np.matrix(Y.values)
    # print("\n========== X ==========\n", X )
    # print("\n========== Y ==========\n", Y )

    model = linear_model.LinearRegression()
    model.fit(X, Y)

    x = np.array(X[:, 1].A1)
    f = model.predict(X).flatten()


    plt.figure("Sklearn")
    plt.title("Predicted Profit vs. Population Size")
    plt.xlabel("Population")
    plt.ylabel("Profit")
    plt.plot(x, f, "r", label="Prediction")
    plt.scatter(data.Population, data.Profit, label = "Traning Data")
    plt.legend(loc=2)
    plt.subplot


    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)