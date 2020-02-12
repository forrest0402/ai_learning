# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 13/2/2020
"""


class Window:
    def exit(self):
        print("exit")


class MenuItem:
    def click(self):
        self.command()


if __name__ == "__main__":
    window = Window()
    menu_item = MenuItem()
    menu_item.command = window.exit
    menu_item.click()
