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


def main_func(argv):
    X, y_true = make_blobs(n_samples = 700,
                           centers = 5,
                           cluster_std = 0.5,
                           random_state = 2019)

    # X = X[:, ::-1]

    gmm = GMM(n_components=3).fit(X)
    labels = gmm.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")
    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)