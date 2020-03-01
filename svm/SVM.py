# -*- coding: utf-8 -*-

"""
@Ref: Tensorflow machine learning cookbook 2017
@Author: xiezizhe
@Date: 13/2/2020
"""
import unittest

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf

import utils.vector_ops as vector_ops
from utils.DataLoader import DataLoader


class LinearSVM(tf.keras.Model):
    """
    SVM implemention: y = W @ x+b
    """

    def __init__(self, num_feature=2, c=0.01):
        super(LinearSVM, self).__init__()
        self.W = tf.Variable(tf.random.truncated_normal(shape=[num_feature, 1]), dtype=tf.float32)
        self.b = tf.Variable(tf.random.truncated_normal(shape=[1, 1]), dtype=tf.float32)
        self.alpha = tf.constant([c], dtype=tf.float32)

    def call(self, x, **kwargs):
        model_output = tf.add(tf.matmul(x, self.W), self.b)
        return model_output

    def loss(self, y, model_output):
        l2_norm = tf.nn.l2_loss(self.W)
        classification_term = tf.reduce_mean(tf.maximum(0., 1.0 - tf.multiply(model_output, y)))
        loss = tf.add(classification_term, self.alpha * l2_norm)
        return loss

    def accu(self, y, model_output):
        prediction = tf.sign(model_output)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))
        return accuracy

    def params(self):
        return self.W.numpy(), self.b.numpy()


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

    def call(self, x, xi=None, **kwargs):
        """

        :param x:
        :param xi:
        :param kwargs:
        :return: shape (x.shape[0],landmark.shape[0])
        """
        kernel = vector_ops.square_dist(x, xi, lambda v: tf.exp(-self.gamma * tf.abs(v)))
        return kernel

    def loss(self, y, kernel):
        aa = tf.matmul(tf.transpose(self.alpha), self.alpha)
        yy = tf.matmul(y, tf.transpose(y))
        loss = tf.reduce_sum(tf.multiply(kernel, tf.multiply(aa, yy))) - self.C * tf.reduce_sum(self.alpha)
        if np.isnan(loss):
            print('wrong')
        return loss

    def accu(self, y, yi, pred_kernel):
        model_output = tf.matmul(tf.transpose(yi) * self.alpha, tf.transpose(pred_kernel))
        prediction = tf.sign(model_output - tf.reduce_mean(model_output))
        if y is not None:
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y)), tf.float32))
            return accuracy, tf.reshape(prediction, (-1,))
        return np.nan, tf.reshape(prediction, (-1,))


class AMSVM(tf.keras.Model):
    """
    SVM implemention: y = Wx+b
    """

    def __init__(self, num_classes, num_feature=2, c=0.01):
        super(AMSVM, self).__init__()
        self.W = tf.Variable(tf.random.truncated_normal(shape=[num_feature, num_classes]), dtype=tf.float32)
        self.b = tf.Variable(tf.random.truncated_normal(shape=[1, num_classes]), dtype=tf.float32)
        self.c = tf.constant([c], dtype=tf.float32)

    def call(self, x, **kwargs):
        model_output = tf.add(tf.matmul(x, self.W), self.b)
        return model_output

    def loss(self, y, model_output):
        l2_norm = tf.nn.l2_loss(self.W)
        classification_term = tf.reduce_mean(tf.maximum(0., 1.0 - tf.multiply(model_output, y)))
        loss = tf.add(classification_term, self.c * l2_norm)
        return loss

    def accu(self, y, model_output):
        prediction = tf.sign(model_output)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))
        return accuracy

    def params(self):
        return self.W.numpy(), self.b.numpy()


class TestSVM(unittest.TestCase):
    def setUp(self) -> None:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices is not None and len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.data_loader = DataLoader()

    def test_linear(self):
        (x_train, y_train), (x_test, y_test) = self.data_loader.loadIris1(0.8)
        svm = LinearSVM(num_feature=2)
        svm.compile(optimizer=tf.optimizers.SGD(0.01), loss=svm.loss, metrics=[svm.accu])
        svm.fit(x_train, y_train, batch_size=64, epochs=400, verbose=0)

        results = svm.evaluate(x_test, y_test)
        print("test result: ", results, svm.params())

        self.assertGreater(results[1], 0.9)

        a = float(-svm.W[0] / svm.W[1])
        xx = np.linspace(-2.5, 2.5)
        yy = a * xx - float(svm.b / svm.W[1])

        self.data_loader.plot1((0.0, 10.0),
                               (float(-svm.b.numpy() / svm.W.numpy()[1]),
                                float((-svm.b.numpy() - 10 * svm.W.numpy()[0]) / svm.W.numpy()[1])),
                               color='black').show()

    def test_gaussian(self):
        def draw(x_vals, y_vals, show=True):
            class1_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == 1]
            class1_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == 1]
            class2_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == -1]
            class2_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == -1]
            if show:
                plt.plot(class1_x, class1_y, 'ro', label='I. setosa')
                plt.plot(class2_x, class2_y, 'kx', label='I. versicolor')
                # plt.plot(class3_x, class3_y, 'gv', label='I. virginica')
                plt.title('Gaussian SVM Results on Iris Data')
                plt.xlabel('Pedal Length')
                plt.ylabel('Sepal Width')
                plt.legend(loc='lower right')
                plt.show()
            return class1_x, class1_y, class2_x, class2_y

        (x_vals, y_vals) = sklearn.datasets.make_circles(n_samples=3000, factor=.5, noise=.1)
        y_vals = np.array([1.0 if y == 1.0 else -1.0 for y in y_vals], dtype=np.float)

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
        test_dataset = gen_dataset(x_test, y_test, 5)

        # train
        def train_step(x_sample, y_sample):
            with tf.GradientTape() as tape:
                pred_kernel = svm(x_sample, x_sample)
                loss = svm.loss(y_sample, pred_kernel)
                accu, _ = svm.accu(y_sample, y_sample, pred_kernel)
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

        # test
        rand_index = np.random.choice(len(x_vals), size=batch_size)
        rand_x = x_vals[rand_index]
        rand_y = tf.convert_to_tensor(np.transpose([y_vals[rand_index]]), dtype=tf.float32)
        accus = []
        for (batch, (x, y)) in enumerate(test_dataset):
            pred_kernel = svm(x, rand_x)
            accu, _ = svm.accu(y, rand_y, pred_kernel)
            accus.append(accu)
        print("test accuracy: {}".format(np.mean(accus)))
        self.assertGreater(np.mean(accus), 0.8)
        # plot results
        x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
        y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        output_kernel = svm(grid_points, rand_x)
        _, predictions = svm.accu(None, rand_y, output_kernel)
        grid_predictions = tf.reshape(predictions, xx.shape)

        # Plot points and grid
        class1_x, class1_y, class2_x, class2_y = draw(x_vals, y_vals, False)
        plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
        plt.plot(class1_x, class1_y, 'ro', label='Class 1')
        plt.plot(class2_x, class2_y, 'kx', label='Class -1')
        plt.title('Gaussian SVM Results')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='lower right')
        plt.ylim([-1.5, 1.5])
        plt.xlim([-1.5, 1.5])
        plt.show()

    def test_amsvm(self):
        (class1_x, class1_y), (class2_x, class2_y) = self.data_loader.loadIris1(0.8)
        svm = AMSVM(num_classes=2, num_feature=2)
        svm.compile(optimizer=tf.optimizers.SGD(0.01), loss=svm.loss, metrics=[svm.accu])
        svm.fit(x_train, y_train, batch_size=64, epochs=400, verbose=0)

        results = svm.evaluate(x_test, y_test)
        print("test result: ", results, svm.params())

        self.assertGreater(results[1], 0.9)

        a = float(-svm.W[0] / svm.W[1])
        xx = np.linspace(-2.5, 2.5)
        yy = a * xx - float(svm.b / svm.W[1])

        self.data_loader.plot1((0.0, 10.0),
                               (float(-svm.b.numpy() / svm.W.numpy()[1]),
                                float((-svm.b.numpy() - 10 * svm.W.numpy()[0]) / svm.W.numpy()[1])),
                               color='black').show()


if __name__ == "__main__":
    print(tf.__version__)
    print("hello, world")
