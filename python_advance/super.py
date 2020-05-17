# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 18/5/2020 上午12:39
"""


class DictionaryDB(object):

    def __init__(self, data):
        self.data = data

    def __getattribute__(self, item):
        data = super().__getattribute__('data')
        return data[item]


if __name__ == "__main__":
    d = DictionaryDB({'fool': 'abc'})
    print(d.fool)
