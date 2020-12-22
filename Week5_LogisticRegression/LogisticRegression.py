# -*- coding: utf-8 -*-
# @Time    : 2020/12/22 15:53
# @Author  : lowkeyway
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

def main_func(argv):
    data = pd.read_csv("ex2data1.txt", names=['exam1', 'exam2', 'admitted'])
    # print(data.head())

    sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))
    sns.lmplot('exam1', 'exam2', hue='admitted', data=data,height=12, fit_reg=False, scatter_kws={"s":50})
    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)