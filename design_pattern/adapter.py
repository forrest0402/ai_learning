# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 13/2/2020
"""

import datetime


class AgeCal:
    def __init__(self, birthday):
        self.year, self.month, self.day = (int(x) for x in birthday.split('-'))

    def cal(self, date):
        y, m, d = (int(x) for x in date.split('-'))
        if (m, d) < (self.month, self.day):
            return y - self.year - 1
        return y - self.year


class DateAgeCal:
    def _str_date(self, date):
        return date.strftime("%Y-%m-%d")

    def __init__(self, birthday):
        self.cal = AgeCal(self._str_date(birthday))

    def get_age(self, date):
        date = self._str_date(date)
        return self.cal.cal(date)


if __name__ == "__main__":
    d = DateAgeCal(datetime.date(1992, 4, 1))
    print(d.get_age(datetime.date.today()))
