# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 13/2/2020
"""
import weakref
import gc


class CarModel:
    _models = weakref.WeakValueDictionary()

    def __new__(cls, name, *args, **kwargs):
        model = cls._models.get(name)
        if not model:
            model = super().__new__(cls)
            cls._models[name] = model
        return model

    def __init__(self, name, color='yellow'):
        if not hasattr(self, 'initted'):
            self.name = name
            self.initted = True


if __name__ == "__main__":
    m1 = CarModel("Benz", color='red')
    print(id(m1))
    m2 = CarModel("Benz")
    print(id(m2))
    del m1
    del m2
    gc.collect()
    m3 = CarModel("Benz")
    print(id(m3))
