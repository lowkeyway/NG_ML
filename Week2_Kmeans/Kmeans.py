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

def cost(data, centroids, C):
    m = data.shape[0]

    expandCWithCentroids = centroids[C]
    distances = np.apply_along_axis(func1d=np.linalg.norm,
                                    axis=1,
                                    arr=data.values - expandCWithCentroids)

    return distances.sum()/m

def kMeansIter(data, k, epoch=100, tol=0.0001):
    # 1. 随机选择k个点作为Centroids
    centroids = randomInit(data, k)
    costProgress = []

    for i in range(epoch):
        print("running epoch {}".format(i))
        # 2. 增加一列C，根据数据data中的每个像素相对于centroids的距离，进行归类打标签成离自己最近的那个centroids的值(0, 1, ... , k)
        C = assignCluster(data, centroids)
        # 3. 根据归类后的数据，重新计算每个类的mean值，定义为新的centroids
        centroids = newCentroids(data, C)
        # 4. 计算每个簇所有点到其中心点的平均距离记做cost
        costProgress.append(cost(data, centroids, C))

        # 5. 终止触发，缩小的比例小于一定的值就认为已经足够优秀了
        if len(costProgress) > 1:
            if(np.abs(costProgress[-1] - costProgress[-2])) / costProgress[-1] < tol:
                break

        dataWithC = combineDataC(data, C)
        # fig, ax = plt.subplots(figsize=(12, 8))
        # ax.set_title("Index " + str(i))
        sns.lmplot("X1", "X2", hue="C", data=dataWithC, fit_reg=False)
        sns.lmplot("X1", "X2", data=pd.DataFrame(centroids, columns=["X1", "X2"]), markers="X", fit_reg=False)
        ax = plt.gca()
        ax.set_title("Index " + str(i))
        # plt.title("Index " + str(i))
        plt.show()

    return C, centroids, costProgress[-1]

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
    #
    # initCentroids = randomInit(data, 3)
    # print(initCentroids)

    # X = np.array([1, 1])
    # fig, ax = plt.subplots(figsize=(6, 4))
    # ax.scatter(x=initCentroids[:, 0], y=initCentroids[:, 1])
    #
    # for i, node in enumerate(initCentroids):
    #     ax.annotate("[{}: ({}, {})]" .format(i, node[0], node[1]), node)
    #
    # ax.scatter(X[0], X[1], marker='x', s=200)
    #
    # col = findYourCluster(X, initCentroids)
    # print(col)
    #
    # C = assignCluster(data, initCentroids)
    # print("C = \n", C)
    #
    # dataWithC = combineDataC(data, C)
    # print(dataWithC)
    # sns.lmplot("X1", "X2", hue="C", data=dataWithC, fit_reg=False)
    # plt.show()
    #
    # newCent = newCentroids(data, C)
    # print("newCent = \n", newCent)


    finalC, finalCentroid, finnalCost = kMeansIter(data, 3)


if __name__ == '__main__':
    main_func(sys.argv)