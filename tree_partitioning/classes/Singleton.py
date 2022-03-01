#!/usr/bin/env ipython
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            up = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = up

        return cls._instances[cls]

    def clear(cls):
        try:
            del Singleton._instances[cls]
        except KeyError:
            pass
