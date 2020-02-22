# -*- coding: utf-8 -*-

"""
@Ref: Tensorflow machine learning cookbook 2017
@Author: xiezizhe
@Date: 13/2/2020
"""
import tensorflow as tf
import numpy as np
from utils.DataLoader import DataLoader
import unittest
import sklearn
import matplotlib.pyplot as plt
import utils.vector_ops as vector_ops


class LinearSVM(tf.keras.Model):
    """
    SVM implemention: y = W @ x+b
    """

    def __init__(self, num_feature=2, C=1.0):
        super(LinearSVM, self).__init__()
        self.W = tf.Variable(tf.random.truncated_normal(shape=[num_feature, 1]), dtype=tf.float32)
        self.b = tf.Variable(tf.random.truncated_normal(shape=[1, 1]), dtype=tf.float32)
        self.alpha = tf.constant([C], dtype=tf.float32)

    def call(self, x, **kwargs):
        model_output = tf.add(tf.matmul(x, self.W), self.b)
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


class PolynomialKernelSVM(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(PolynomialKernelSVM, self).__init__(*args, **kwargs)


class GaussianKernelSVM(tf.keras.Model):
    """
    Gaussian Kernel: K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
    性质 1. 若RBF核中尺度参数σ趋于0，则拉格朗日乘子向量的所有分量都大于0，即全部样本点都是支持向量。
    性质1 说明，对于任意给定的训练集，只要σ>0且充分小，RBF核SVM必定可对所有训练样本正确分类，这很容易造成‘过学习’的情况。
    性质 2. 当σ趋于无穷时，SVM的判别函数为一常函数，其推广能力或对新样本的正确分类能力为零，即把所有样本点判为同一类。
    """

    def __init__(self, batch_size=256, gamma=50, c=1, *args, **kwargs):
        super(GaussianKernelSVM, self).__init__(*args, **kwargs)
        self.gamma = tf.constant(gamma, dtype=tf.float32)  # 2 / sigma^2, smaller gamma, larger radius
        self.alpha = tf.Variable(tf.random.truncated_normal(shape=[1, batch_size]), dtype=tf.float32)
        self.C = tf.constant([c], dtype=tf.float32)
        self.batch_size = batch_size

    def call(self, x, landmark=None, **kwargs):
        kernel = vector_ops.square_dist(x, landmark, lambda v: tf.exp(-self.gamma * tf.abs(v)))
        model_output = tf.matmul(self.alpha, kernel)
        return model_output, kernel

    def loss(self, y, kernel):
        aa = tf.matmul(tf.transpose(self.alpha), self.alpha)
        yy = tf.matmul(y, tf.transpose(y))
        loss = tf.reduce_sum(tf.multiply(kernel, tf.multiply(aa, yy))) - self.C * tf.reduce_sum(self.alpha)
        if np.isnan(loss):
            print('wrong')
        return loss

    def accu(self, y, model_output):
        model_output = tf.multiply(tf.transpose(y), model_output)
        prediction = tf.sign(model_output - tf.reduce_mean(model_output))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y)), tf.float32))
        return accuracy


class NonLinearSVM(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, x_data, **kwargs):
        model_output = tf.add(tf.matmul(x_data, self.W), self.b)
        return model_output

    def loss(self, y, model_output):
        pass

    def accu(self, y, model_output):
        pass


class TestSVM(unittest.TestCase):
    def setUp(self) -> None:
        self.data_loader = DataLoader()

    def test_linear(self):
        (x_train, y_train), (x_test, y_test) = self.data_loader.loadIris1(0.8)
        svm = LinearSVM()
        svm.compile(optimizer=tf.optimizers.SGD(0.01), loss=svm.loss, metrics=[svm.accu])
        svm.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0)

        results = svm.evaluate(x_test, y_test)
        print("test result: ", results)

        a = float(-svm.W[0] / svm.W[1])
        xx = np.linspace(-2.5, 2.5)
        yy = a * xx - float(svm.b / svm.W[1])

        self.data_loader.plot1((0.0, 10.0),
                               (float(-svm.b.numpy() / svm.W.numpy()[1]),
                                float((-svm.b.numpy() - 10 * svm.W.numpy()[0]) / svm.W.numpy()[1])),
                               color='black').show()

    def test_gaussian(self):
        def draw(x_vals, y_vals):
            class1_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == 1]
            class1_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == 1]
            class2_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == -1]
            class2_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == -1]
            plt.plot(class1_x, class1_y, 'ro', label='I. setosa')
            plt.plot(class2_x, class2_y, 'kx', label='I. versicolor')
            # plt.plot(class3_x, class3_y, 'gv', label='I. virginica')
            plt.title('Gaussian SVM Results on Iris Data')
            plt.xlabel('Pedal Length')
            plt.ylabel('Sepal Width')
            plt.legend(loc='lower right')
            # plt.ylim([-0.5, 3.0])
            # plt.xlim([3.5, 8.5])
            plt.show()

        (x_vals, y_vals) = sklearn.datasets.make_circles(n_samples=3000, factor=.5, noise=.1)
        y_vals = np.array([1.0 if y == 1 else -1.0 for y in y_vals], dtype=np.float)

        split_ratio = 0.9
        x_train, y_train = x_vals[0: int(len(x_vals) * split_ratio)], y_vals[0: int(len(y_vals) * split_ratio)]
        x_test, y_test = x_vals[int(len(x_vals) * split_ratio):], y_vals[int(len(y_vals) * split_ratio):]
        draw(x_train, y_train)
        draw(x_test, y_test)

        def gen_dataset(x, y, batch_size):
            x, y = tf.cast(x, dtype=tf.float32), tf.reshape(tf.cast(y, dtype=tf.float32), shape=(-1, 1))
            return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size, drop_remainder=True)

        batch_size = 256
        epochs = 300
        svm = GaussianKernelSVM(batch_size=batch_size)
        optimizer = tf.keras.optimizers.SGD(0.001)
        train_dataset = gen_dataset(x_train, y_train, batch_size)
        test_dataset = gen_dataset(x_test, y_test, batch_size)

        def train_step(x_sample, y_sample):
            with tf.GradientTape() as tape:
                predictions, kernel = svm(x_sample)
                loss = svm.loss(y_sample, kernel)
                accu = svm.accu(y_sample, predictions)
            gradients = tape.gradient(loss, svm.trainable_variables)  # had to indent this!
            optimizer.apply_gradients(zip(gradients, svm.trainable_variables))
            return loss, accu

        for epoch in range(epochs):
            accus, losses = [], []
            for (batch, (x, y)) in enumerate(train_dataset):
                loss, accu = train_step(x_sample=x, y_sample=y)
                accus.append(accu.numpy())
                losses.append(loss.numpy())
            print("Epoch: {}, accu: {}, loss: {}".format(epoch, np.mean(accus), np.mean(losses)))

        accus = []
        for (batch, (x, y)) in enumerate(test_dataset):
            predictions, kernel = svm(x)
            accu = svm.accu(y, predictions)
            accus.append(accu)
        print("test accuracy: {}".format(np.mean(accus)))


if __name__ == "__main__":
    print(tf.__version__)
    print("hello, world")
