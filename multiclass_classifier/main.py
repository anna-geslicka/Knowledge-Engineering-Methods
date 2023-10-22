import argparse
import copy

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions
from perceptron import Perceptron
from reglog import LogisticRegressionGD

iris = datasets.load_iris()
x = iris.data[:, [2, 3]]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

data_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
# classifier_type = {0: 'Perceptron', 1: 'Logistic regression'}


class Classifier:
    def __init__(self, set_cl, vers_cl):
        self.set_cl = set_cl
        self.vers_cl = vers_cl

    def predict(self, val):
        return np.where(self.set_cl.predict(val) == 1, 0, np.where(self.vers_cl.predict(val) == 1, 2, 1))


def perc1():
    my_x_train = copy.deepcopy(x_train)
    my_y_train = copy.deepcopy(y_train)
    my_y_train[y_train != 0] = -1
    my_y_train[y_train == 0] = 1
    perceptron = Perceptron(alpha=0.1, n_iter=1000)
    print(type(my_x_train))
    perceptron.fit(my_x_train, my_y_train)
    return perceptron


def perc2():
    my_x_train = copy.deepcopy(x_train)
    my_y_train = copy.deepcopy(y_train)
    my_y_train[y_train != 2] = -1
    my_y_train[y_train == 2] = 1
    perceptron = Perceptron(alpha=0.1, n_iter=1000)
    perceptron.fit(my_x_train, my_y_train)
    return perceptron


def reglog1():
    my_x_train = copy.deepcopy(x_train)
    my_y_train = copy.deepcopy(y_train)
    my_y_train[y_train != 0] = 0
    my_y_train[y_train == 0] = 1

    #    my_y_train[y_train == -1] = 0

    lrgd = LogisticRegressionGD(alpha=0.05, n_iter=1000, random_state=1)
    lrgd.fit(my_x_train, my_y_train)
    return lrgd


def reglog2():
    my_x_train = copy.deepcopy(x_train)
    my_y_train = copy.deepcopy(y_train)
    my_y_train[y_train != 2] = 0
    my_y_train[y_train == 2] = 1
    #    my_y_train[y_train == -1] = 0

    lrgd = LogisticRegressionGD(alpha=0.05, n_iter=1000, random_state=1)
    lrgd.fit(my_x_train, my_y_train)
    return lrgd


def parse_arguments():
    parser = argparse.ArgumentParser(description='This is a script for multi-class classification.')
    args = parser.parse_args()


def main(args):
    perceptron = Classifier(perc1(), perc2())
    lrgd = Classifier(reglog1(), reglog2())
    # plot_decision_regions(x=x_test, y=y_test, classifier=perceptron, label_mapping=d)
    plot_decision_regions(x=x_test, y=y_test, classifier=lrgd, label_mapping=data_names)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main(parse_arguments())
