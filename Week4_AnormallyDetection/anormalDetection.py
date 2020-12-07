# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 13:50
# @Author  : lowkeyway
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

def select_threshold(X, Xval, yval):
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    pval = multi_normal.pdf(Xval)

    epsilon = np.linspace(np.min(pval), np.max(pval), num=10000)

    fs = []
    for e in epsilon:
        y_pred = (pval <= e).astype("int")
        fs.append(f1_score(yval, y_pred))

    argmax_fs = np.argmax(fs)

    return epsilon[argmax_fs], fs[argmax_fs]

def main_func(argv):
    mat = sio.loadmat("./data/ex8data1.mat")
    print("mat.keys() = \n", mat.keys())
    print("mat = \n", mat)
    X = mat.get("X")
    print("X = \n", X)

    Xval, Xtest, Yval, Ytest = train_test_split(mat.get("Xval"),
                                                mat.get("yval").ravel(),
                                                test_size=0.5)

    print("Xval = \n", Xval)
    print("Xtest = \n", Xtest)
    print("Yval = \n", Yval)
    print("Ytest = \n", Ytest)
    sns.set(context="notebook", style="white", palette=sns.color_palette("RdBu"))
    sns.regplot("Latency", "Throughput",
                data = pd.DataFrame(X, columns=["Latency", "Throughput"]),
                fit_reg=False,
                scatter_kws={"s":20, "alpha":0.5})
    plt.show()


    mu = X.mean(axis=0)
    print("mu = \n", mu)

    cov = np.cov(X.T)
    print(cov)

    multi_normal = stats.multivariate_normal(mu, cov)

    x, y = np.mgrid[0:30:0.01, 0:30:0.01]
    pos = np.dstack((x, y))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.contourf(x, y, multi_normal.pdf(pos), cmap="Blues")

    sns.regplot("Latency", "Throughput",
                data = pd.DataFrame(X, columns=["Latency", "Throughput"]),
                fit_reg=False,
                ax = ax,
                scatter_kws={
                    "s":10,
                    "alpha":0.4
                })
    plt.show()

    e, fs = select_threshold(X, Xval, Yval)
    print("Best epsilon: {}\nBest F-score on validation data: {}" .format(e, fs))

if __name__ == '__main__':
    main_func(sys.argv)