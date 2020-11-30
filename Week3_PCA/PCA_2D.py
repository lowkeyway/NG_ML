# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 15:11
# @Author  : lowkeyway
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import pandas as pd

def normalize(X):
    X_copy = X.copy()
    m, n = X_copy.shape

    for col in range(n):
        X_copy[:, col] = (X_copy[:, col] - X_copy[:, col].mean()) / X_copy[:, col].std()

    return X_copy

def covarianceMatrix(X):
    m = X.shape[0]

    return (X.T @ X) / m

def PCA(X):
    X_normal = normalize(X)
    Sigma = covarianceMatrix(X_normal)

    U, S, V = np.linalg.svd(Sigma)
    return U, S, V


def projectData(X, U, k):
    m, n = X.shape

    if k > n:
        raise ValueError("K shold be lower than dimension of n")

    return X @ U[:, :k]

def recoverData(Z, U):
    m, n = Z.shape

    if n > U.shape[0]:
        raise ValueError("Z dimension is >= U, you shold recover from lower dimension to higher")

    return Z @ U[:, :n].T

def main_func(argv):
    mat = sio.loadmat("./data/ex7data1.mat")
    data = mat.get("X")
    pdData = pd.DataFrame(data, columns=["X1", "X2"])

    norData = normalize(data)
    pdNorData = pd.DataFrame(norData, columns=["X1", "X2"])

    U, S, V = PCA(norData)
    Z = projectData(norData, U, 1)

    recData = recoverData(Z, U)
    pdRecData = pd.DataFrame(recData, columns=["X1", "X2"])

    # print("mat = \n", mat)
    # print("data = \n", data)

    # sns.lmplot("X", "Y", data=data, fit_reg=False)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

    axes[0, 0].set_title("Original Data")
    sns.regplot("X1","X2", data=pdData, ax=axes[0, 0], fit_reg=False)

    axes[0, 1].set_title("Normalize Data")
    sns.regplot("X1", "X2", data=pdNorData, fit_reg=False, ax=axes[0, 1])

    axes[1, 0].set_title("Temp & Recover Data")
    sns.regplot("X1", "X2", data=pdRecData, fit_reg=False, ax=axes[1, 0])

    axes[1, 1].set_title("Z dimension")
    axes[1, 1].set_xlabel("Z")
    sns.rugplot(Z, ax=axes[1, 1])


    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)