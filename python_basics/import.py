# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 5/2/2020 下午9:51
"""
from python_basics.import_package import Conn
import python_basics.import_package.Database
from python_basics.import_package import Database
from python_basics.import_package.Database import Connector

# Note that relative imports are based on the name of the current module.
# Since the name of the main module is always "__main__",
# modules intended for use as the main module of a Python application must always use absolute imports.
# from .import_package import Database as Db

if __name__ == "__main__":
    c1 = python_basics.import_package.Database.Connector()
    c2 = Database.Connector()
    c3 = Connector()
    print(Conn()._Connector__variable)
