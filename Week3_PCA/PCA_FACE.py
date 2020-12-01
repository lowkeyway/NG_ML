# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 17:27
# @Author  : lowkeyway
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.io as sio

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

def plot_n_image(X, n, title=" "):
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n, :]
    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size, sharey=True, sharex=True, figsize=(8, 8), num=title)

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

def main_func(argv):
    mat = sio.loadmat("./data/ex7faces.mat")
    # print(mat.get("X"))

    X = np.array([x.reshape((32, 32)).T.reshape(1024) for x in mat.get("X")])
    print("X.size = ", X.size)

    plot_n_image(X, n=64, title="X")

    U, S, V = PCA(X)
    Z = projectData(X, U, k=100)

    plot_n_image(Z, n=64, title="Z")
    print("Z.size = ", Z.size)

    X_recover = recoverData(Z, U)
    plot_n_image(X_recover, n=64, title="X_recover")
    print("X_recover.size = ", X_recover.size)

    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)