# -*- coding: utf-8 -*-
# @Time    : 2020/11/17 17:26
# @Author  : lowkeyway
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



def main_func(argv):
    x = np.random.uniform(-3, 3, size=100)
    x = np.sort(x)
    print("x = \n", x)
    y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, 100)

    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    print("X = \n", X)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    print("X_poly = \n", X_poly)

    linReg = LinearRegression()
    linReg.fit(X_poly, Y)
    print("linReg.coef_, linReg.intercept_ = \n", linReg.coef_, linReg.intercept_ )
    # yPredict = linReg.predict(X_poly)
    yPredict = np.dot(X_poly, linReg.coef_.T) + linReg.intercept_

    plt.figure("Polynomial")
    plt.title("2 + x + x^2/2")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(x, y, label="Data")
    # plt.plot(np.sort(x), yPredict[np.argsort(x)], color="r", label = "Predict")
    plt.plot(x, yPredict, color="r")
    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)