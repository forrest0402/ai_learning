# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 7/2/2020 上午12:53
"""


class Invalid(Exception):
    pass


def funny_division(num):
    try:
        if num == 13:
            raise Invalid("unlucky number")
        return 100 / num
    except (ZeroDivisionError, TypeError):
        return 'Enter a number other than zero'
    except ValueError:
        return "not 13"
    except Exception as e:
        return e.args
    finally:
        print("finally")


if __name__ == "__main__":
    print(funny_division(13))
