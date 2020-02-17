# -*- coding: utf-8 -*-

"""
@Ref: https://www.cvxpy.org/examples/machine_learning/svm.html
@Author: xiezizhe
@Date: 17/2/2020 下午2:45
"""
import cvxpy as cp
from utils.DataLoader import DataLoader
import numpy as np

if __name__ == "__main__":
    data_loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.loadIris1(0.8)

    n = 2
    m = len(x_train)

    W = cp.Variable((n, 1))
    b = cp.Variable()
    loss = cp.sum(cp.pos(1 - cp.multiply(np.reshape(y_train.numpy(), (m, 1)), x_train.numpy() @ W + b)))
    reg = cp.norm(W, 1)
    lambd = cp.Parameter(nonneg=True)
    prob = cp.Problem(cp.Minimize(loss / m + lambd * reg))
    lambd.value = 0.1
    prob.solve()

    print("{} * w + {}".format(W.value, b.value))
    data_loader.plot1((0.0, 10.0),
                      (float(-b.value / W.value[1]),
                       float((-b.value - 10 * W.value[0]) / W.value[1])), color='black').show()

    print('hello, world')
