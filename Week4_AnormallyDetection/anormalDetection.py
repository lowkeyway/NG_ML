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

"""
1. use CV data to find the best epsilon
2. use all data (training + validation) to create model
3. do the prediction on test data
"""

def select_threshold(X, Xval, yval):
    """
    use CV data to find the best epsilon
    Returns:
        e: best epsilon with the higest f-score
        f-score: such best f-score
        about what is f1: https://www.cnblogs.com/178mz/p/8558435.html
    """
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

def predict(X, Xval, e, Xtest, yTest):
    """
    with optimal epsilon, combine X, Xval and predict Xtest.
    Returns:
        multi_normal: multivariate normal model
        y_pred: prediction of test data
    """
    Xdata = np.concatenate((X, Xval), axis=0)

    mu = Xdata.mean(axis=0)
    cov = np.cov(Xdata.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    # calculate probability of test data
    pval = multi_normal.pdf(Xtest)
    y_pred = (pval <= e).astype("int")

    # precision recall f1-score三列分别为各个类别的精确度/召回率及 F1 值．
    print(classification_report(yTest, y_pred))

    return multi_normal, y_pred

def main_func(argv):
    mat = sio.loadmat("./data/ex8data1.mat")
    print("mat.keys() = \n", mat.keys())
    print("mat = \n", mat)
    X = mat.get("X")
    print("X = \n", X)

    # Get Train data and Test data from Xval and Yval
    Xval, Xtest, Yval, Ytest = train_test_split(mat.get("Xval"),
                                                mat.get("yval").ravel(),
                                                test_size=0.5)

    print("Xval = \n", Xval)
    print("Xtest = \n", Xtest)
    print("Yval = \n", Yval)
    print("Ytest = \n", Ytest)

    sns.set(context="notebook", style="white", palette=sns.color_palette("RdBu"))

    # scatter_kws is some config setting about scatter plot.
    sns.regplot("Latency", "Throughput",
                data = pd.DataFrame(X, columns=["Latency", "Throughput"]),
                fit_reg=False,
                scatter_kws={"s":20, "alpha":0.5})

    # plt.show()


    # mu = X.mean(axis=0)
    # print("mu = \n", mu)
    #
    # # Estimate a covariance matrix, given data and weights.
    # cov = np.cov(X.T)
    # print(cov)
    #
    # # Build a multi gaussian distribution.
    # multi_normal = stats.multivariate_normal(mu, cov)
    #
    # x, y = np.mgrid[0:30:0.01, 0:30:0.01]
    # pos = np.dstack((x, y))
    #
    # fig, ax = plt.subplots(figsize=(12, 8))
    #
    # # 绘制等高线
    # ax.contourf(x, y, multi_normal.pdf(pos), cmap="Blues")
    #
    # sns.regplot("Latency", "Throughput",
    #             data = pd.DataFrame(X, columns=["Latency", "Throughput"]),
    #             fit_reg=False,
    #             ax = ax,
    #             scatter_kws={
    #                 "s":10,
    #                 "alpha":0.4
    #             })
    # plt.show()

    e, fs = select_threshold(X, Xval, Yval)
    print("Best epsilon: {}\nBest F-score on validation data: {}" .format(e, fs))

    multi_normal, y_pred = predict(X, Xval, e, Xtest, Ytest)

    data = pd.DataFrame(Xtest, columns=["Latency", "Throughput"])
    data["y_pred"] = y_pred

    # create a grid for graphing
    x, y = np.mgrid[0:30:0.01, 0:30:0.01]
    pos = np.dstack((x, y))

    fig, ax = plt.subplots(figsize=(12, 8))

    # plot probabiliy density
    ax.contourf(x, y, multi_normal.pdf(pos), cmap="Blues")

    # plot original Xval point
    sns.regplot("Latency", "Throughput",
                data = data,
                fit_reg=False,
                ax = ax,
                scatter_kws={
                    "s" : 10,
                    "alpha" : 0.4
                })


    # mark the predicted anamoly of CV data, We should have a test set for this...
    anamoly_data = data[data["y_pred"] == 1]
    ax.scatter(anamoly_data["Latency"], anamoly_data["Throughput"], marker="x", s=50)

    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)