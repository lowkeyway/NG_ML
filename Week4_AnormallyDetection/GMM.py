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
    else:
        ax.scatter(X[:, 0], X[:, 1], s=4, zorder=2)

    ax.axis("equal")
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def main_func(argv):
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
    X_stretched = np.dot(X, rng.random((2, 2)))
    # X_stretched = X
    gmm = GMM(n_components=4, covariance_type="full", random_state=42)
    plot_gmm(gmm, X_stretched)

    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)