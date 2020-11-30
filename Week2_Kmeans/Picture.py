# -*- coding: utf-8 -*-
# @Time    : 2020/11/29 21:12
# @Author  : lowkeyway
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from skimage import io
from sklearn.cluster import KMeans


def main_func(argv):
    pic = io.imread("./data/bird_small.png")
    # io.imshow(pic)
    # io.show()
    print("pic.shape = \n" ,pic.shape)
    picRow, picCol, RGB = pic.shape[:]
    data = pic.reshape(picRow * picCol, RGB)
    print("data.shape = \n" ,data.shape)

    # model = KMeans(n_clusters=16, n_init=100)
    model = KMeans(n_clusters=6)
    model.fit(data)

    centroids = model.cluster_centers_
    print(centroids.shape)

    C = model.predict(data)
    print("C.shape", C.shape)

    print("centroids[C].shape = \n", centroids[C].shape)

    compressedPic = centroids[C].astype(int).reshape((picRow, picCol, RGB))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(pic)
    ax[1].imshow(compressedPic)
    plt.show()



if __name__ == '__main__':
    main_func(sys.argv)