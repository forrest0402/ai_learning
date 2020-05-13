# -*- coding: utf-8 -*-

"""
if exam instance doesn't have writing_grade attribute, python would access exam class to finding class attribute
@Author: xiezizhe
@Date: 14/5/2020 上午1:08
"""
from weakref import WeakKeyDictionary


class Grade(object):

    def __init__(self):
        self._values = WeakKeyDictionary()

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self._values[instance]

    def __set__(self, instance, value):
        value = max(value, 0)
        value = min(value, 100)
        self._values[instance] = value


class Exam(object):
    writing_grade = Grade()
    math_grade = Grade()


if __name__ == "__main__":
    e, b = Exam(), Exam()
    e.writing_grade, b.writing_grade = -5, 60
    e.math_grade, b.math_grade = 200, 95
    print(e.writing_grade)
    print(e.math_grade)
    print(b.writing_grade)
    print(b.math_grade)
