# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 9/2/2020
"""
import json


def filter(sequence, condition):
    for s in sequence:
        if condition(s):
            yield s


def get_pages(*links):
    pass


class Options:
    default_options = {
        'port': 21,
        'host': 'localhost'
    }

    def __init__(self, **kwargs):
        self.option = dict(Options.default_options)
        self.option.update(**kwargs)


if __name__ == "__main__":
    l = [1, 2, 3]
    print(len(l))  # len(l) == l.__len__()
    print(json.dumps(list(reversed(l))))
    for (i, item) in enumerate(l):
        print(i, item)

    # zip
    list_one = [(1, 'a'), (2, 'b'), (3, 'c')]
    print(list(zip(*list_one)))  # [(1, 2, 3), ('a', 'b', 'c')]

    list_two = (v for v in list_one if v[0] > 1)  # generate an iterator
    print("generator: ", list(list_two))

    list_two = filter(list_one, lambda x: x[0] > 1)
    print("filter: ", list(list_two))
