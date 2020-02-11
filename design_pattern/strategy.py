# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 10/2/2020
"""


class Method:
    def __call__(self, *args, **kwargs):
        print('hello, world')


if __name__ == "__main__":
    m = Method()
    m()
