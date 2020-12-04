# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 15:31
# @Author  : lowkeyway
import sys
import numpy as np


def main_func(argv):
    x = np.mgrid[0:3, 0:3]
    print("x = \n", x)

    y = np.dstack(x)
    print("y = \n", y)

    z = np.eye(3)
    zCov = np.cov(z)
    print("z = \n", z)
    print("zCov = \n", zCov)



if __name__ == '__main__':
    main_func(sys.argv)