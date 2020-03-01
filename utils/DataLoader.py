# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 13/2/2020
"""
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class DataLoader:
    def __load__(self, dataset_name):
        if dataset_name == 'iris1':
            iris = datasets.load_iris()
            x = np.array([[x[0], x[3]] for x in iris.data])
            y = np.array([1 if y == 0 else -1 for y in iris.target])
            return x, y

    def loadIris1(self, ratio=0.8):
        x, y = self.__load__('iris1')
        # plt.scatter(x=list(zip(*x))[0], y=list(zip(*x))[1])
        # plt.show()
        train_indices = np.random.choice(len(x), round(len(x) * ratio), replace=False)
        test_indices = np.array(list(set(range(len(x))) - set(train_indices)))
        x_train = tf.cast(x[train_indices], dtype=tf.float32)
        x_test = tf.cast(x[test_indices], dtype=tf.float32)
        y_train = tf.cast(y[train_indices], dtype=tf.float32)
        y_test = tf.cast(y[test_indices], dtype=tf.float32)
        return (x_train, y_train), (x_test, y_test)

    def loadIris2(self, ratio=0.8):
        iris = datasets.load_iris()
        x_vals = np.array([[x[0], x[3]] for x in iris.data])
        y_vals1 = np.array([1 if y == 0 else -1 for y in iris.target])
        y_vals2 = np.array([1 if y == 1 else -1 for y in iris.target])
        y_vals3 = np.array([1 if y == 2 else -1 for y in iris.target])
        y_vals = np.array([y_vals1, y_vals2, y_vals3])
        class1_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 0]
        class1_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 0]
        class2_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 1]
        class2_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 1]
        class3_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 2]
        class3_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 2]
        return (class1_x, class1_y), (class2_x, class2_y), (class3_x, class3_y)

    def plot1(self, d1, d2, color='red'):
        x, y = self.__load__('iris1')
        x1 = [d for i, d in enumerate(x) if y[i] == 1]
        x2 = [d for i, d in enumerate(x) if y[i] == -1]
        print("I. setosa={}; Non setosa={}".format(len(x1), len(x2)))
        plt.plot(list(zip(*x1))[1], list(zip(*x1))[0], 'ro', label='I. setosa')
        plt.plot(list(zip(*x2))[1], list(zip(*x2))[0], 'kx', label='Non setosa')

        plt.plot(d2, d1, color=color)
        return plt
