# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 13/2/2020
"""
import unittest


class GenerateNumber:
    def get(self):
        return 1

    def zero(self, n):
        return n / 0


class CheckNumbers(unittest.TestCase):

    def setUp(self):
        self.g = GenerateNumber()

    def test_get(self):
        self.assertEqual(1, self.g.get())

    @unittest.expectedFailure
    def test_get2(self):
        self.assertEqual("1.0", self.g.get())

    def test_zero(self):
        with self.assertRaises(ZeroDivisionError):
            self.g.zero(10)

    def tearDown(self):
        self.g = None


if __name__ == "__main__":
    unittest.main()
