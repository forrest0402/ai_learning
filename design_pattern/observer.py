# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 10/2/2020
"""


class MessageCenter:
    def __init__(self):
        self.observers = []
        self._message = None

    def attach(self, *os):
        for o in os:
            self.observers.append(o)

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, msg):
        self._message = msg
        self._update()

    def _update(self):
        for observer in self.observers:
            observer()


class Consumer:
    def __init__(self, name, message_center):
        self.name = name
        self.message_center = message_center

    def __call__(self, *args, **kwargs):
        print(self.name, self.message_center.message)


if __name__ == "__main__":
    mc = MessageCenter()
    c1 = Consumer('c1', mc)
    c2 = Consumer('c2', mc)
    mc.attach(c1, c2)
    mc.message = 'Hello, world'
