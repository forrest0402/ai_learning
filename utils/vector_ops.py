# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 22/2/2020 下午11:42
"""
import tensorflow as tf
import numpy as np


def square_dist(x, y=None, f=None):
    """

    :param f:
    :param x: shape (n,m), x=[x1,x2,...xn]
    :param y: shape (n',m)
    :return: if x==y, then output Y where yij = (xi-xj)^2
    """
    if y is None:
        y = x
    if not isinstance(x, tf.Tensor):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
    if not isinstance(y, tf.Tensor):
        y = tf.convert_to_tensor(y, dtype=tf.float32)
    sq_x = tf.reduce_sum(tf.square(x), 1)  # shape (n,)
    sq_y = tf.reduce_sum(tf.square(y), 1)
    sq_x = tf.expand_dims(sq_x, -1)  # or tf.reshape(dist, [-1,1]) shape (n,1)
    sq_y = tf.expand_dims(sq_y, -1)
    xx = 2. * tf.matmul(x, tf.transpose(y))  # (n,n)ij=2xixj
    sq_dists = tf.add(tf.subtract(sq_x, xx), tf.transpose(sq_y))
    if f is not None:
        sq_dists = f(sq_dists)
    return sq_dists


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(tf.__version__)
    x = np.random.randint(-5, 5, (3, 2))
    print(x)
    print(square_dist(x))
