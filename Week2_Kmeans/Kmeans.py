# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 19:12
# @Author  : lowkeyway
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import seaborn as sns


def main_func(argv):
    mat = sio.loadmat("./data/ex7data2.mat")
    data = pd.DataFrame(mat.get("X"), columns=["X1", "X2"])
    print(data.head())

    sns.set(context="notebook", style="white")
    sns.lmplot("X1", "X2", data=data, fit_reg=False)
    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)