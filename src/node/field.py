import math
import sympy as sp


class AbstractField:
    def __init__(self, name, d_name, tpe, sizes, has_device_ptr):
        self.name = name
        self.d_name = d_name
        self.tpe = tpe
        self.sizes = sizes
        self.has_device_ptr = has_device_ptr

    def totalSize(self):
        return math.prod(self.sizes)

    def linearizeIt(self, iterators):
        for i, it in enumerate(iterators):
            if 0 == i:
                linearized = it
            else:
                stride = math.prod(self.sizes[0: i])
                linearized = stride * it + linearized

        return linearized

    def access(self, iterators):
        return FieldAccess(self, self.linearizeIt(iterators), True)

    def d_access(self, iterators):
        return FieldAccess(self, self.linearizeIt(iterators), False)


class FieldAccess(sp.Symbol):
    def __new__(cls, field, index, host, description=''):
        obj = sp.Symbol.__new__(cls, f'{field.name if host else field.d_name}[{index}]')  # .replace('[', '_').replace(']', '_'))
        obj.description = description
        return obj

    def __init__(self, field, index, host):
        self.field = field
        self.index = index
        self.host = host
