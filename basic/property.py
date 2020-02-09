# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 9/2/2020
"""


class Color:
    def __init__(self, name, rgb):
        self._name = name
        self._rgb = rgb

    def _set_name(self, name):
        if name is None or len(name) < 1:
            raise Exception("Invalid name")
        self._name = name

    def _get_name(self):
        return self._name

    @property
    def rgb(self):
        return self._rgb

    @rgb.setter
    def rgb(self, value):
        self._rgb = value

    name = property(_get_name, _set_name)


if __name__ == "__main__":
    c = Color('red', '#fff000')
    print(c.name, c.rgb)
