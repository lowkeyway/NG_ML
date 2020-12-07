# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 15:31
# @Author  : lowkeyway
import sys
import numpy as np


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

    X = np.array([[1, 5, 6],
                  [4, 3, 9],
                  [4, 2, 9],
                  [4, 7, 2]])
    print("X = \n", X)

    x = X[0:2]
    y = X[2:4]

    print("x = \n", x)
    print("y = \n", y)

    print("np.cov(X) = \n", np.cov(X))

    print("np.cov(x, y) = \n", np.cov(x, y))


if __name__ == '__main__':
    main_func(sys.argv)