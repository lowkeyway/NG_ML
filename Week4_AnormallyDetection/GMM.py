# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 10:44
# @Author  : lowkeyway
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

SIZE = 10

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

# 给定的位置和协方差画一个椭圆
def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()

    #将协方差转换为主轴
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, heigh = 2 * np.sqrt(s)
    else:
        angle = 0
        width, heigh = 2 * np.sqrt(covariance)


    # 画出椭圆
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * heigh, angle, **kwargs))

#画图
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)

    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s = 4, cmap="viridis", zorder=2)
        pass
    else:
        ax.scatter(X[:, 0], X[:, 1], s=4, zorder=2)

    ax.axis("equal")
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def main_func(argv):


    D1, G1 = Gaussian_Distribution(N=2, M=SIZE * SIZE, m=1000, sigma=100)
    D2, G2 = Gaussian_Distribution(N=2, M=SIZE * SIZE, m=5000, sigma=200)
    D3, G3 = Gaussian_Distribution(N=2, M=SIZE * SIZE, m=10000, sigma=300)

    data = np.concatenate((D1, D2, D3), axis=0)

    X, y_true = make_blobs(n_samples = 700,
                           centers = 4,
                           cluster_std = 0.5,
                           random_state = 2019)

    X = X[:, ::-1]
    #
    # gmm = GMM(n_components=3).fit(X)
    # labels = gmm.predict(X)
    #
    # plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")

    rng = np.random.RandomState(13)
    # X_stretched = np.dot(X, rng.random((2, 2)))
    X_stretched = data
    # plt.scatter(X_stretched[:, 0], X_stretched[:, 1], s=4)
    gmm = GMM(n_components=3, covariance_type="full", random_state=42)
    plot_gmm(gmm, X_stretched)

    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)