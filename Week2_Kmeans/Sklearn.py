# -*- coding: utf-8 -*-
# @Time    : 2020/11/29 12:15
# @Author  : lowkeyway
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def combineDataC(data, C):
    dataWithC = data.copy()
    dataWithC["C"] = C
    return dataWithC

def main_func(argv):
    mat = sio.loadmat("./data/ex7data2.mat")
    data = pd.DataFrame(mat.get("X"), columns=["X1", "X2"])

    skKmeans = KMeans(n_clusters=3)
    skKmeans.fit(data)

    skC = skKmeans.predict(data)

    dataWithC = combineDataC(data, skC)
    sns.lmplot("X1", "X2", hue="C", data=dataWithC, fit_reg=False)

    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)