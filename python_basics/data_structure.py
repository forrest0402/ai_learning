# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 9/2/2020
"""
from collections import namedtuple
from collections import defaultdict
from collections import KeysView, ItemsView, ValuesView

num_items = 0


def _namedtuple():
    Color = namedtuple("Color", "name rgb")
    c = Color("red", "#fff000")
    print(c)


def _defaultdict():
    freq = defaultdict(int)
    freq['a'] += 1
    freq['b'] = 10
    print(freq)

    def count():
        global num_items
        num_items += 1
        return num_items, []

    d = defaultdict(count)
    d['a'][1].append('hello')
    d['b'][1].append('world')
    print(d)


def _sort_list():
    l = [{'name': 'zizhe', 'value': 10}, {'name': 'zeng', 'value': 1}, {'name': 'x', 'value': 5}]
    l.sort(key=lambda x: x['value'], reverse=True)
    print(l)


class MemoryDict(dict):
    """
    dir(dict)
    help(dict.__getitem__)
    """
    def __new__(*args, **kwargs):
        new_dict = dict.__new__(*args, **kwargs)
        new_dict.ordered_keys = []
        return new_dict

    def __setitem__(self, key, value):
        '''
        self[key]=value
        :param key:
        :param value:
        :return:
        '''
        if key not in self.ordered_keys:
            self.ordered_keys.append(key)
        super().__setitem__(key, value)

    def setdefault(self, key, value):
        if key not in self.ordered_keys:
            self.ordered_keys.append(key)
        return super().setdefault(key, value)

    def keys(self):
        return KeysView(self)

    def values(self):
        return ValuesView(self)

    def items(self):
        return ItemsView(self)

    def __iter__(self):
        self.ordered_keys.sort(key=lambda x: self.__getitem__(x))
        return self.ordered_keys.__iter__()


if __name__ == "__main__":
    _namedtuple()
    _defaultdict()
    _sort_list()

    mDict = MemoryDict()
    mDict['a'] = 5
    mDict['c'] = 10
    mDict['b'] = 6
    mDict['d'] = 7
    mDict.setdefault('a', 100)
    mDict.setdefault('e', 0)
    for d in mDict.items():
        print(d)
