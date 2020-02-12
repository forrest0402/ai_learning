# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 11/2/2020
"""


class Only:
    _singleton = None

    def __new__(cls, *args, **kwargs):
        if not cls._singleton:
            cls._singleton = super(Only, cls).__new__(cls, *args, **kwargs)
        return cls._singleton


if __name__ == "__main__":
    o1, o2 = Only(), Only()
    print(o1 == o2)
