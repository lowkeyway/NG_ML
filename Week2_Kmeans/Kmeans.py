# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 19:12
# @Author  : lowkeyway
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import seaborn as sns

def combineDataC(data, C):
    dataWithC = data.copy()
    dataWithC["C"] = C
    return dataWithC

def randomInit(data, k):
    return data.sample(k).values

def findYourCluster(x, centroids):
    distances = np.apply_along_axis(func1d=np.linalg.norm,
                                    axis=1,
                                    arr=centroids-x)

    return np.argmin(distances)

def assignCluster(data, centroids):
    return np.apply_along_axis(lambda x: findYourCluster(x, centroids),
                               axis=1,
                               arr=data.values)

def newCentroids(data, C):
    dataWithC = combineDataC(data, C)
    # print("dataWithC = \n", dataWithC)
    data = dataWithC.groupby("C", as_index=False).mean().sort_values(by="C").drop("C", axis=1).values
    return data

def main_func(argv):
    mat = sio.loadmat("./data/ex7data2.mat")
    data = pd.DataFrame(mat.get("X"), columns=["X1", "X2"])
    print(data.head())

    # sns.set(context="notebook", style="white")
    # sns.lmplot("X1", "X2", data=data, fit_reg=False)
    # plt.show()

    initCentroids = randomInit(data, 3)
    print(initCentroids)

    # X = np.array([1, 1])
    # fig, ax = plt.subplots(figsize=(6, 4))
    # ax.scatter(x=initCentroids[:, 0], y=initCentroids[:, 1])
    #
    # for i, node in enumerate(initCentroids):
    #     ax.annotate("[{}: ({}, {})]" .format(i, node[0], node[1]), node)
    #
    # ax.scatter(X[0], X[1], marker='x', s=200)

    # col = findYourCluster(X, initCentroids)
    # print(col)

    C = assignCluster(data, initCentroids)
    print("C = \n", C)

    # dataWithC = combineDataC(data, C)
    # print(dataWithC)
    # sns.lmplot("X1", "X2", hue="C", data=dataWithC, fit_reg=False)
    # plt.show()

    newCent = newCentroids(data, C)
    print("newCent = \n", newCent)

if __name__ == '__main__':
    main_func(sys.argv)