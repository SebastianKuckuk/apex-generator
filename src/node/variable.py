import sympy as sp


class Variable(sp.Symbol):
    def __new__(cls, name, description=''):
        obj = sp.Symbol.__new__(cls, name)
        obj.description = description
        return obj

    def __init__(self, name, tpe):
        self.name = name
        self.tpe = tpe

    def __str__(self):
        return f'{self.name}'

    def decl(self):
        return f'{self.tpe} {self.name}'
