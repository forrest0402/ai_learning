# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 1/2/2020 下午10:12
"""
from sklearn.preprocessing import MinMaxScaler
import sklearn

if __name__ == "__main__":
    X = [[89, 7921], [72, 5184], [94, 8836], [69, 4761]]
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    print(X)
    X = sklearn.preprocessing.normalize(X)
    print(X)
