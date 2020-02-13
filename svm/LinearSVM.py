# -*- coding: utf-8 -*-

"""
@Ref: https://medium.com/cs-note/tensorflow-ch4-support-vector-machines-c9ad18878c76
@Author: xiezizhe
@Date: 13/2/2020
"""
import tensorflow as tf
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from utils.DataLoader import DataLoader


class LinearSVM(tf.keras.Model):
    def __init__(self):
        super(LinearSVM, self).__init__()
        self.W = tf.Variable(tf.random.truncated_normal(shape=[2, 1]), dtype=tf.float32)
        self.b = tf.Variable(tf.random.truncated_normal(shape=[1, 1]), dtype=tf.float32)
        self.alpha = tf.constant([0.1], dtype=tf.float32)

    def call(self, x_data, **kwargs):
        model_output = tf.add(tf.matmul(x_data, self.W), self.b)
        return model_output

    def loss(self, y, model_output):
        l2_norm = tf.nn.l2_loss(self.W)
        classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y))))
        loss = tf.add(classification_term, tf.multiply(self.alpha, l2_norm))
        return loss

    def accu(self, y, model_output):
        prediction = tf.sign(model_output)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))
        return accuracy

    def params(self):
        return self.W.numpy(), self.b.numpy()


if __name__ == "__main__":
    print(tf.__version__)
    data_loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.loadIris1(0.8)
    svm = LinearSVM()
    svm.compile(optimizer=tf.optimizers.SGD(0.01), loss=svm.loss, metrics=[svm.accu])
    svm.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0)

    results = svm.evaluate(x_test, y_test)
    print("test result: ", results)

    a = float(-svm.W[0] / svm.W[1])
    xx = np.linspace(-2.5, 2.5)
    yy = a * xx - float(svm.b / svm.W[1])

    data_loader.plot1((0.0, 10.0),
                      (float(-svm.b.numpy() / svm.W.numpy()[1]),
                       float((-svm.b.numpy() - 10 * svm.W.numpy()[0]) / svm.W.numpy()[1])), color='black').show()

    # data_loader.plot1(xx, yy).show()

    print("hello, world")
