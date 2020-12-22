# -*- coding: utf-8 -*-
# @Time    : 2020/12/17 14:37
# @Author  : lowkeyway
import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV


def main_func(argv):
    # load data
    data = load_iris()
    X = data['data']
    Y = data['target']

    # split arrays or matrices into random train and test subsets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=0)

    # how k influences the model
    f1array = []
    f1array_train = []
    for i in range(1, len(X_train) + 1):
        # model train
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, Y_train)

        # model predict
        Y_pred = model.predict(X_test)

        # model evaluation
        f1array.append(f1_score(Y_test, Y_pred, average='macro'))

        Y_pred_train = model.predict(X_train)
        f1array_train.append(f1_score(Y_train, Y_pred_train, average='macro'))

    for i in range(0, len(f1array)):
        print("k=", str(i + 1), ",f1_score:" + str(f1array[i]))

    print("the largest f1_score is " + str(np.max(f1array)))

    for i in range(0, len(f1array_train)):
        print("k=", str(i + 1), ",f1_score:" + str(f1array_train[i]))

    print("the largest f1_score for train is " + str(np.max(f1array_train)))

    plt.plot(f1array, label="test", color='blue', linestyle='-')
    plt.plot(f1array_train, label="train", color='yellow', linestyle='-')
    plt.show()

    # how to choose k
    model = KNeighborsClassifier()
    param_grid = [
        {'n_neighbors': list(range(1, 50))}]
    grid_search = GridSearchCV(model, param_grid, cv=3,
                               scoring='f1_macro')
    grid_search.fit(X_train, Y_train)
    print("the best params：", grid_search.best_params_)

    # model predict
    Y_pred = grid_search.predict(X_test)
    print("the f1_score in test set is:", str(f1_score(Y_test, Y_pred, average='macro')))


if __name__ == '__main__':
    main_func(sys.argv)