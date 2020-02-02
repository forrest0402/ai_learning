# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 1/2/2020 下午11:40
"""
import tensorflow as tf
import numpy as np


class LinearModel:
    def __init__(self):
        self.Weight = tf.Variable(10, dtype=tf.float32)
        self.Bias = tf.Variable(10, dtype=tf.float32)

    def __call__(self, x):
        return self.Weight * x + self.Bias

    def variables(self):
        return self.Weight, self.Bias


if __name__ == "__main__":
    num_epochs = 100
    X = tf.constant([[1, 3], [1, 1], [1, 2], [1, 4]], dtype=tf.float32)
    Y = tf.constant([[9], [5], [7], [11]], dtype=tf.float32)
    r = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)
    print(r)
    X = tf.constant([[3], [1], [2], [4]], dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(500).repeat(num_epochs).batch(4)
    iterator = dataset.__iter__()
    optimizer = tf.keras.optimizers.SGD()
    loss_object = tf.keras.losses.MeanSquaredError()
    model = LinearModel()
    for epoch in range(40):
        losses = []
        for (batch, (x, y)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = model(x)
                loss = tf.reduce_mean(loss_object(y, predictions))
                losses.append(loss.numpy())
                gradients = tape.gradient(loss, model.variables())
                optimizer.apply_gradients(zip(gradients, model.variables()))
        print("{}:{} = {}".format(epoch, np.mean(losses), model.variables()))
