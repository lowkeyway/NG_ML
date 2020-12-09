# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 15:31
# @Author  : lowkeyway
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

SIZE = 100


def Gaussian_Distribution(N=2, M=1000, m=0, sigma=1):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本标准差

    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * (sigma * sigma)  # 协方差矩阵，每个维度的方差都为 sigma

    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)

    return data, Gaussian

def main_func(argv):
    # x = np.mgrid[0:3, 0:3]
    # print("x = \n", x)
    #
    # y = np.dstack(x)
    # print("y = \n", y)
    #
    # z = np.eye(3)
    # zCov = np.cov(z)
    # print("z = \n", z)
    # print("zCov = \n", zCov)

    # X = np.array([[1, 5, 6],
    #               [4, 3, 9],
    #               [4, 2, 9],
    #               [4, 7, 2]])
    # print("X = \n", X)
    #
    # x = X[0:2]
    # y = X[2:4]
    #
    # print("x = \n", x)
    # print("y = \n", y)
    #
    # print("np.cov(X) = \n", np.cov(X))
    #
    # print("np.cov(x, y) = \n", np.cov(x, y))


    # X = np.array([[1, 5, 6],
    #               [4, 3, 9],
    #               [4, 2, 9],
    #               [4, 7, 2]])
    # print("X = \n", X)
    # print("X.shape = \n", X.shape)
    #
    # Y = np.moveaxis(X, 0, -1)
    # print("Y = \n", Y)
    # print("Y.shape = \n", Y.shape)

    Y = np.zeros(SIZE)
    # Y = np.zeros((1, SIZE))
    # print(Y)

    # plt.figure("0D")
    # D0, G0 = Gaussian_Distribution(N=1, M=SIZE, m=0, sigma=1)
    # plt.scatter(D0.tolist(), Y.tolist(), marker="x", c="g")
    # print(D0.tolist())
    # x = np.linspace(-3, 3, 1000)
    # y0 = G0.pdf(x)
    # plt.plot(x, y0, c="r", label="N(0, 1)")
    # plt.legend(loc='best')
    #
    # plt.show()
    # return

    plt.figure("1D")
    # G1 = np.random.normal(loc=1000, scale=100, size=SIZE)
    # G2 = np.random.normal(loc=5000, scale=500, size=SIZE)
    # G3 = np.random.normal(loc=10000, scale=1000, size=SIZE)
    D1, G1 = Gaussian_Distribution(N=1, M=SIZE, m=1000, sigma=100)
    D2, G2 = Gaussian_Distribution(N=1, M=SIZE, m=5000, sigma=500)
    D3, G3 = Gaussian_Distribution(N=1, M=SIZE, m=10000, sigma=1000)

    # fig, ax = plt.subplots(figsize=(12, 8))
    plt.scatter(D1.tolist(), Y.tolist(), marker="x", c="r")
    plt.scatter(D2.tolist(), Y.tolist(), marker="x", c="r")
    plt.scatter(D3.tolist(), Y.tolist(), marker="x", c="r")
    plt.xlabel("Wages")

    x = np.linspace(0, 20000, 10000)
    y1 = G1.pdf(x)
    y2 = G2.pdf(x)
    y3 = G3.pdf(x)

    plt.plot(x, y1, c="r", label="N(1000, 100)")
    plt.plot(x, y2, c="b", label="N(5000, 500)")
    plt.plot(x, y3, c="g", label="N(10000, 1000)")

    plt.legend(loc='best')

    plt.figure("2D")
    D1, G1 = Gaussian_Distribution(N=2, M=SIZE * SIZE, m=1000, sigma=100)
    D2, G2 = Gaussian_Distribution(N=2, M=SIZE * SIZE, m=5000, sigma=200)
    D3, G3 = Gaussian_Distribution(N=2, M=SIZE * SIZE, m=10000, sigma=300)

    plt.scatter(D1[:, 0], D1[:, 1])
    plt.scatter(D2[:, 0], D2[:, 1])
    plt.scatter(D3[:, 0], D3[:, 1])

    M = 100
    # 生成二维网格平面
    X, Y = np.meshgrid(np.linspace(0, 20000, M), np.linspace(0, 20000, M))
    # 二维坐标数据
    d = np.dstack([X, Y])
    # 计算二维联合高斯概率
    Z1 = G1.pdf(d).reshape(M, M)
    Z2 = G2.pdf(d).reshape(M, M)
    Z3 = G3.pdf(d).reshape(M, M)

    # 二元高斯概率分布图
    fig = plt.figure("3D")
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap='seismic', alpha=0.8)
    ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap='seismic', alpha=0.8)
    ax.plot_surface(X, Y, Z3, rstride=1, cstride=1, cmap='seismic', alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)