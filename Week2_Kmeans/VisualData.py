# -*- coding: utf-8 -*-
# @Time    : 2020/11/24 13:43
# @Author  : lowkeyway
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import seaborn as sns

def main_func(argv):
    mat = sio.loadmat("./data/ex7data1.mat")
    mat2 = sio.loadmat("./data/ex7data2.mat")
    # print(mat)

    data = pd.DataFrame(mat.get("X"), columns=["X1", "X2"])
    data2 = pd.DataFrame(mat2.get("X"), columns=["X1", "X2"])
    # print(data)

    X = data.values[:, 0]
    Y = data.values[:, 1]

    plt.figure("Figure")
    plt.title("Title")
    plt.scatter(X, Y)

    sns.set(context="notebook", style="darkgrid")
    sns.lmplot("X1", "X2", data = data2, fit_reg=False)

    plt.show()



if __name__ == '__main__':
    main_func(sys.argv)