# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 22/2/2020 下午11:42
"""
import tensorflow as tf
import numpy as np


def square_dist(x):
    """

    :param x: shape (n,m), x=[x1,x2,...xn]
    :return: Y where yij = (xi-xj)^2
    """
    if not isinstance(x, tf.Tensor):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
    sq = tf.reduce_sum(tf.square(x), 1)  # shape (n,)
    sq = tf.expand_dims(sq, -1)  # or tf.reshape(dist, [-1,1]) shape (n,1)
    xx = 2. * tf.matmul(x, tf.transpose(x))  # (n,n)ij=2xixj
    sq_dists = tf.add(tf.subtract(sq, xx), tf.transpose(sq))
    return sq_dists


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(tf.__version__)
    x = np.random.randint(1, 5, (3, 2))
    print(x)
    print(square_dist(x))
