def inherit_docs(cls):
    cls.__doc__ = cls.__bases__[0].__doc__
    return cls
