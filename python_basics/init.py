# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 13/2/2020
"""


class Parent:
    def __new__(cls, *args, **kwargs):
        return "New Parent"

    def __str__(self):
        return "Parent"


class Child:
    def __new__(cls):
        return Parent()

    def __init__(self):
        print("Inside init")


class Child2:
    def __new__(cls):
        pass

    def __init__(self):
        print("Hello, world")


if __name__ == "__main__":
    print(Child())  # Parent
    print(Child2())
