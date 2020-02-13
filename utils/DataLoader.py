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

    def plot1(self, d1, d2, color='red'):
        x, y = self.__load__('iris1')
        x1 = [d for i, d in enumerate(x) if y[i] == 1]
        x2 = [d for i, d in enumerate(x) if y[i] == -1]
        print("I. setosa={}; Non setosa={}".format(len(x1), len(x2)))
        plt.plot(list(zip(*x1))[1], list(zip(*x1))[0], 'ro', label='I. setosa')
        plt.plot(list(zip(*x2))[1], list(zip(*x2))[0], 'kx', label='Non setosa')

        plt.plot(d2, d1, color=color)
        return plt
