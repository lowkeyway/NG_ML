# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 10:46
# @Author  : lowkeyway
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

def main_func(argv):
    # 加载鸢尾花数据
    iris = datasets.load_iris()
    print("iris = \n", iris)
    # 只采用样本数据的前两个feature，生成X和Y
    X = iris.data[:, :2]
    Y = iris.target
    h = .02  # 网格中的步长
    # 新建模型，设置C参数为1e5，并进行训练
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)
    # 绘制决策边界。为此我们将为网格 [x_min, x_max]x[y_min, y_max] 中的每个点分配一个颜色。
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    # 将结果放入彩色图中
    Z = Z.reshape(xx.shape)
    print("Z(Predict) = \n", Z)
    plt.figure(1, figsize=(12, 8))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    # 将训练点也同样放入彩色图中
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()


if __name__ == '__main__':
    main_func(sys.argv)