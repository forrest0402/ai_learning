# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 10/2/2020
"""
import datetime
import time


class Printer:
    def print(self, msg):
        time.sleep(1)
        print(msg)


class TimePrinter:
    def __init__(self, printer):
        self.printer = printer

    def print(self, msg):
        start_time = datetime.datetime.now()
        self.printer.print(msg)
        elapsed_time = datetime.datetime.now() - start_time
        print('{} ms'.format(elapsed_time.total_seconds() * 1000))


def time_calls(func):
    def wrapper(*args, **kwargs):
        now = time.time()
        return_value = func(*args, **kwargs)
        print('Executed {0} in {1} ms'.format(func.__name__, (time.time() - now) * 1000))
        return return_value

    return wrapper


@time_calls
def print_msg(msg):
    time.sleep(1)
    print(msg)


if __name__ == "__main__":
    p = Printer()
    tp = TimePrinter(p)
    tp.print("hello, world")

    tp = time_calls(p.print)
    tp("hello, world")

    print_msg("Hello, world")
