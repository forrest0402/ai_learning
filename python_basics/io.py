# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 13/2/2020
"""


class StringJoiner(list):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.result = "".join(self)


if __name__ == "__main__":
    # context manager, invoke __enter__, __exit__
    try:
        with open('filename', 'r', encoding='utf-8') as file:
            for line in file:
                print(line, end=' ')
    except FileNotFoundError as f:
        print(f.args[1])

    with StringJoiner() as j:
        for i in range(10):
            j.append(str(i))

    print(j.result)
