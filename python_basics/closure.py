# -*- coding: utf-8 -*-

"""
Python functions are first-class citizens.
it is a nested funcion
it has access to a free variable in outer scope
it is returned from the enclosing function
@Author: xiezizhe
@Date: 27/4/2020 上午2:02
"""
from typing import List


def sort1(n: List, group: List):
    found = False

    def helper(x):
        if x in group:
            found = True
            return (0, x)
        return (1, x)

    n.sort(key=helper)
    return found


def sort2(n: List, group: List):
    found = False

    def helper(x):
        nonlocal found
        if x in group:
            found = True
            return (0, x)
        return (1, x)

    n.sort(key=helper)
    return found


class Sorter(object):

    def __init__(self, g):
        self.g = g
        self.found = False

    def __call__(self, x):
        if x in self.g:
            self.found = True
            return 0, x
        return 1, x


def sort3(n: List, group: List):
    s = Sorter(group)
    n.sort(key=s)
    return s.found


def main():
    n = [3, 4, 5, 1, 2]
    print(sort1(n, [5]))
    print(n)

    print(sort2(n, [5]))
    print(n)

    # Python closures can be an alternate solution to small classes.
    print(sort3(n, [5]))
    print(n)


if __name__ == "__main__":
    main()
